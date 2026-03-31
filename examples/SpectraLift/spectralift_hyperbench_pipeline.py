"""
SpectraLift + HyperBench integration example.

This file adapts the original SpectraLift implementation to HyperBench's
`run_pipeline(...)` interface.

Key integration ideas
---------------------
1. HyperBench provides the degraded inputs:
   - HR_MSI with shape (H, W, c)
   - LR_HSI with shape (H/r, W/r, C)
   - srf    with shape (c, C)

2. This file is responsible only for model-side logic:
   - preparing TensorFlow tensors
   - training the SpectraLift spectral MLP
   - running inference
   - returning the predicted HR HSI

3. HyperBench remains responsible for benchmark orchestration:
   - generating degradations
   - looping over experiment cases
   - computing reconstruction metrics (RMSE, PSNR, SSIM, SAM, ...)
   - writing benchmark rows to CSV

4. SpectraLift already performs some tasks internally, so HyperBench does not
   need to repeat them here:
   - NumPy -> TensorFlow conversion
   - TensorFlow -> NumPy conversion
   - output clipping to [0, 1]

5. This integration returns:
   - prediction
   - stats dictionary

   HyperBench can then add the returned stats to its CSV output.

Notes
-----
- This file keeps the overall author-side logic intact.
- The main behavioral change is that model statistics are RETURNED instead of
  only being printed.
- HyperBench will compute image-quality metrics itself from the returned
  prediction, so those metrics should not be computed here.
"""

################################
# Hyperbench imports

from pathlib import Path
import csv
import sys

import cv2
import numpy as np

from hyperbench import (
    BenchmarkConfig,
    DegradationSpec,
    PipelineAdapter,
    load_benchmark_config,
    load_hsi,
    normalize_image,
    print_data_stats,
    run_benchmark,
)

################################

################################
# SpectraLift imports

import scipy.io as sio
from tqdm import tqdm
import os
import math
import time
import io as iot
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
################################

################################
# SpectraLift implementation code from author's github (replaces the demo method we had before)

def numpy_to_tf(np_array):
    """Convert a NumPy array to a TensorFlow tensor."""
    return tf.constant(np_array, dtype=tf.float32)


def tf_to_numpy(tf_tensor):
    """Convert a TensorFlow tensor to a NumPy array."""
    return tf_tensor.numpy()


def apply_srf_tf(hsi, srf):
    """
    Apply an SRF to an HSI in TensorFlow.

    Parameters
    ----------
    hsi : tf.Tensor
        Hyperspectral image of shape (h, w, C)
    srf : tf.Tensor
        Spectral response function of shape (c, C)

    Returns
    -------
    tf.Tensor
        Low-resolution MSI of shape (h, w, c)
    """
    srf_t = tf.transpose(srf)  # (C, c)
    msi = tf.tensordot(hsi, srf_t, axes=[[-1], [0]])
    return msi


def prepare_inputs(hr_msi, lr_hsi, srf):
    """
    Prepare all inputs required by SpectraLift.

    HyperBench passes NumPy arrays to run_pipeline(...). SpectraLift converts
    them to TensorFlow here and constructs the low-resolution MSI used for
    training the spectral MLP.
    """
    hr_msi = numpy_to_tf(hr_msi)
    lr_hsi = numpy_to_tf(lr_hsi)
    srf = numpy_to_tf(srf)
    lr_msi = apply_srf_tf(lr_hsi, srf)
    return hr_msi, lr_hsi, lr_msi, srf


def get_gpu_memory_mb():
    """
    Return current GPU memory usage in MB for GPU:0.

    If no compatible GPU is available, return NaN rather than failing.
    """
    try:
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        return mem_info["current"] / (1024 ** 2)
    except Exception:
        return float("nan")


def infer_and_analyze_model_performance_tf(model, sample_inputs):
    """
    Analyze model complexity and inference behavior.

    Returns
    -------
    SR_image : tf.Tensor
        Model prediction
    num_params : int
        Number of trainable parameters
    flops : int or float
        Estimated FLOPs
    mem_used : float
        Estimated GPU memory consumption in MB
    inference_time : float
        Inference time in seconds
    """
    input_signature = [
        tf.TensorSpec(shape=inp.shape, dtype=inp.dtype) for inp in sample_inputs
    ]

    @tf.function
    def model_fn(msi):
        return model(msi)

    concrete_func = model_fn.get_concrete_function(*input_signature)
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    try:
        original_stdout = sys.stdout
        sys.stdout = iot.StringIO()

        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts["output"] = "none"
            prof = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                options=opts,
            )
            flops = prof.total_float_ops if prof is not None else float("nan")
    except Exception:
        flops = float("nan")
    finally:
        sys.stdout = original_stdout

    num_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    start_mem = get_gpu_memory_mb()

    start = time.perf_counter()
    SR_image = model(*sample_inputs)
    end = time.perf_counter()
    inference_time = end - start

    end_mem = get_gpu_memory_mb()
    mem_used = end_mem - start_mem if not np.isnan(end_mem) and not np.isnan(start_mem) else float("nan")

    return SR_image, num_params, flops, mem_used, inference_time


def batched_inference(model, hr_msi, batch_size):
    """Run tiled inference over HR_MSI."""
    if isinstance(hr_msi, tf.Tensor):
        hr_msi = hr_msi.numpy()

    H, W, C = hr_msi.shape
    bs = batch_size or max(H, W)
    ys = list(range(0, H, bs))
    xs = list(range(0, W, bs))
    total_batches = len(ys) * len(xs)

    SR = np.zeros((H, W, model.num_outputs), np.float32)

    pbar = tqdm(total=total_batches, desc="Inference")
    for y0 in ys:
        for x0 in xs:
            patch = hr_msi[y0:y0 + bs, x0:x0 + bs]
            patch_tf = tf.constant(patch, tf.float32)
            sr_patch = model(patch_tf).numpy()
            SR[y0:y0 + bs, x0:x0 + bs] = sr_patch
            pbar.update(1)
    pbar.close()

    return SR


class SpectralSR_MLP(Model):
    """SpectraLift spectral MLP."""

    def __init__(self, num_outputs, hidden_size=128):
        super().__init__()
        self.num_outputs = num_outputs

        self.layer1 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer2 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer3 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer4 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer5 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer6 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)

        self.output_layer = Dense(self.num_outputs, activation="linear", dtype=tf.float32)

    def call(self, msi):
        H = tf.shape(msi)[0]
        W = tf.shape(msi)[1]

        x = tf.reshape(msi, [H * W, -1])

        x = self.layer1(x)
        x1 = x
        x = self.layer2(x) + x1

        x2 = x
        x = self.layer3(x)
        x = self.layer4(x) + x2

        x3 = x
        x = self.layer5(x)
        x = self.layer6(x) + x3

        hsi = self.output_layer(x)
        hsi = tf.reshape(hsi, [H, W, self.num_outputs])
        return hsi


def train_spectral_mlp(
    lr_msi,
    lr_hsi,
    epochs=2500,
    lr_schedule="one_cycle",
    init_lr=1e-4,
    max_lr=1e-2,
    final_lr=1e-6,
    min_lr=1e-6,
    num_restarts=1,
    hidden_size=64,
    batch_size=1024,
):
    """Train the SpectraLift spectral MLP."""
    h, w, C = lr_hsi.shape
    model = SpectralSR_MLP(num_outputs=C, hidden_size=hidden_size)

    def get_lr(epoch):
        if lr_schedule == "one_cycle":
            pct_up = 0.3
            if epoch < pct_up * epochs:
                return init_lr + (max_lr - init_lr) * (epoch / (pct_up * epochs))
            return max_lr - (max_lr - final_lr) * (
                (epoch - pct_up * epochs) / ((1 - pct_up) * epochs)
            )
        if lr_schedule == "cosine_restart":
            period = epochs // num_restarts
            cur = epoch % period
            cos_decay = 0.5 * (1 + math.cos(math.pi * cur / period))
            return min_lr + (max_lr - min_lr) * cos_decay
        return init_lr

    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    pbar = tqdm(range(1, epochs + 1), desc="Training SR-MLP", unit="epoch")
    for epoch in pbar:
        lr = get_lr(epoch)
        optimizer.learning_rate.assign(lr)

        if batch_size is None:
            with tf.GradientTape() as tape:
                pred = model(lr_msi)
                loss = loss_fn(lr_hsi, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            pbar.set_postfix(loss=f"{loss.numpy():.4f}", lr=f"{lr:.2e}")
        else:
            epoch_loss = 0.0
            count = 0
            for y0 in range(0, h, batch_size):
                for x0 in range(0, w, batch_size):
                    sub_msi = lr_msi[y0:y0 + batch_size, x0:x0 + batch_size, :]
                    sub_hsi = lr_hsi[y0:y0 + batch_size, x0:x0 + batch_size, :]
                    with tf.GradientTape() as tape:
                        pred = model(sub_msi)
                        loss = loss_fn(sub_hsi, pred)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    epoch_loss += loss.numpy()
                    count += 1

            avg_loss = epoch_loss / count if count else 0.0
            pbar.set_postfix(average_loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

    return model


def run_pipeline(
    HR_MSI,
    LR_HSI,
    srf,
    psf=None,
    metadata=None,
    lr_schedule="one_cycle",
    initial_lr=1e-3,
    max_lr=1e-2,
    final_lr=1e-6,
    min_lr=1e-6,
    num_restarts=2,
    sin_hidden_size=64,
    num_epochs=2500,
    training_batch_size=None,
    inference_batch_size=None,
):
    """
    HyperBench-facing SpectraLift pipeline.

    HyperBench computes reconstruction metrics itself after receiving the
    prediction. This function therefore returns:
    - the prediction
    - model statistics
    """
    training_start = time.perf_counter()

    # SpectraLift performs its own NumPy -> TensorFlow conversion internally.
    hr_msi, lr_hsi, lr_msi, srf_tf = prepare_inputs(HR_MSI, LR_HSI, srf)

    trained_spectral_sr_mlp = train_spectral_mlp(
        lr_msi,
        lr_hsi,
        epochs=num_epochs,
        lr_schedule=lr_schedule,
        init_lr=initial_lr,
        max_lr=max_lr,
        final_lr=final_lr,
        min_lr=min_lr,
        num_restarts=num_restarts,
        hidden_size=sin_hidden_size,
        batch_size=training_batch_size,
    )

    training_end = time.perf_counter()
    training_time = training_end - training_start

    if inference_batch_size is None:
        SR_image_tf, num_params, flops, mem_used, inference_time = (
            infer_and_analyze_model_performance_tf(
                trained_spectral_sr_mlp,
                sample_inputs=[hr_msi],
            )
        )
        SR_image = tf_to_numpy(SR_image_tf)
    else:
        inference_start = time.perf_counter()
        SR_image = batched_inference(
            trained_spectral_sr_mlp,
            HR_MSI,
            inference_batch_size,
        )
        inference_end = time.perf_counter()

        num_params = int(np.sum([np.prod(v.shape) for v in trained_spectral_sr_mlp.trainable_variables]))
        flops = float("nan")
        mem_used = float("nan")
        inference_time = inference_end - inference_start

    # SpectraLift already handles output clipping internally here, which shows
    # that HyperBench can work with models that manage this on their own.
    SR_image = np.clip(SR_image, 0.0, 1.0)

    stats = {
        "framework": "tensorflow",
        "training_time_sec": float(training_time),
        "inference_time_sec": float(inference_time),
        "num_parameters": int(num_params),
        "flops": float(flops) if not isinstance(flops, (int, np.integer)) else int(flops),
        "gpu_memory_mb": float(mem_used),
    }

    return SR_image, stats

class SpectraLiftPipeline:
    def run_pipeline(self, HR_MSI, LR_HSI, srf, psf=None, metadata=None):
        return run_pipeline(HR_MSI, LR_HSI, srf, psf, metadata)
################################

# ------------------------------------------------------------
# Config handling
# ------------------------------------------------------------
def normalize_config(cfg):
    if isinstance(cfg, BenchmarkConfig):
        return cfg

    data = dict(cfg)

    method_name = data.pop("method_name", "SpectraLift")

    if "degradation_specs" in data:
        data["degradation_specs"] = [
            DegradationSpec(**spec) for spec in data["degradation_specs"]
        ]

    config = BenchmarkConfig(**data)
    setattr(config, "_method_name", method_name)
    return config


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_benchmark.py <config.yaml|json>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = normalize_config(load_benchmark_config(config_path))

    # Inspect scene once
    scene = load_hsi(config.scene_path, key=config.scene_key)
    gt_hsi = normalize_image(scene)

    print_data_stats(gt_hsi, name="GT HSI")
    print("Shape:", gt_hsi.shape)
    print()

    # Adapter
    method_name = getattr(config, "_method_name", "SpectraLift")

    adapter = PipelineAdapter(
        pipeline=SpectraLiftPipeline(),
        name=method_name,
        input_backend="numpy",
        output_backend="numpy",
        add_batch_dim=False,
        device="auto",
    )

    print("Running benchmark with config:", config_path)
    print("Method:", method_name)

    results = run_benchmark(adapter, config)

    print("\nRows:", len(results.rows))

    for i, row in enumerate(results.rows):
        print(f"\nRow {i}")
        for k, v in row.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()