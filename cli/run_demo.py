"""
Unified HyperBench runner

This script runs either:
- explicit benchmarks (via degradation_specs), or
- sweep benchmarks (via parameter lists)

The behavior is entirely determined by the config file.
"""

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


# ------------------------------------------------------------
# Demo pipeline (replace this with your real model later)
# ------------------------------------------------------------
def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    target_h = int(HR_MSI.shape[0])
    target_w = int(HR_MSI.shape[1])

    _, _, C_hsi = LR_HSI.shape
    pred = np.empty((target_h, target_w, C_hsi), dtype=np.float32)

    for ch in range(C_hsi):
        pred[:, :, ch] = cv2.resize(
            LR_HSI[:, :, ch],
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return pred, {"framework": "numpy"}


class DemoPipeline:
    def run_pipeline(self, HR_MSI, LR_HSI, srf, psf=None, metadata=None):
        return run_pipeline(HR_MSI, LR_HSI, srf, psf, metadata)


# ------------------------------------------------------------
# Config handling
# ------------------------------------------------------------
def normalize_config(cfg):
    if isinstance(cfg, BenchmarkConfig):
        return cfg

    data = dict(cfg)

    method_name = data.pop("method_name", "DemoMethod")

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
    method_name = getattr(config, "_method_name", "DemoMethod")

    adapter = PipelineAdapter(
        pipeline=DemoPipeline(),
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