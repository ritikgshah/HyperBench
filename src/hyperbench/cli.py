# Author: Ritik Shah

"""Command-line interface for HyperBench."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from hyperbench.adapters import PipelineAdapter
from hyperbench.config import load_benchmark_config
from hyperbench.exceptions import ConfigurationError, HyperBenchError
from hyperbench.utils import print_framework_device_summary
from hyperbench.benchmark import run_benchmark


def _load_module_from_path(module_path: Path):
    if not module_path.exists():
        raise ConfigurationError(
            "Pipeline module file does not exist: '{}'".format(module_path)
        )

    spec = importlib.util.spec_from_file_location(
        "hyperbench_user_pipeline_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ConfigurationError(
            "Failed to create import spec for '{}'".format(module_path)
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_pipeline_object(module, pipeline_name: Optional[str] = None):
    if pipeline_name:
        if not hasattr(module, pipeline_name):
            raise ConfigurationError(
                "Requested pipeline attribute '{}' not found in module '{}'.".format(
                    pipeline_name, module.__name__
                )
            )
        return getattr(module, pipeline_name)

    if hasattr(module, "run_pipeline"):
        return module
    if hasattr(module, "PIPELINE"):
        return getattr(module, "PIPELINE")
    if hasattr(module, "pipeline"):
        return getattr(module, "pipeline")

    raise ConfigurationError(
        "Could not find a pipeline entry point in module '{}'. "
        "Expected one of: run_pipeline, PIPELINE, pipeline.".format(module.__name__)
    )


def _build_adapter_from_args(args):
    module = _load_module_from_path(Path(args.pipeline_module))
    pipeline_obj = _resolve_pipeline_object(module, args.pipeline_name)

    return PipelineAdapter(
        pipeline=pipeline_obj,
        name=args.method_name,
        input_backend=args.input_backend,
        output_backend=args.output_backend,
        add_batch_dim=args.add_batch_dim,
        device=args.device,
        shape_policy=args.shape_policy,
        hr_multiple=args.hr_multiple,
        lr_multiple=args.lr_multiple,
    )


def _add_run_parser(subparsers):
    run_parser = subparsers.add_parser(
        "run",
        help="Run a HyperBench benchmark from a config file.",
    )

    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark config file (.json, .yaml, .yml).",
    )
    run_parser.add_argument(
        "--pipeline-module",
        required=True,
        help="Path to a Python module implementing run_pipeline(...) or exposing a pipeline object.",
    )
    run_parser.add_argument(
        "--pipeline-name",
        default=None,
        help="Optional attribute name inside the module to use as the pipeline object.",
    )
    run_parser.add_argument(
        "--method-name",
        default="pipeline_model",
        help="Method name to appear in HyperBench outputs.",
    )
    run_parser.add_argument(
        "--input-backend",
        default="numpy",
        choices=["numpy", "tensorflow", "torch"],
        help="Input backend to use when passing degraded inputs into the pipeline.",
    )
    run_parser.add_argument(
        "--output-backend",
        default=None,
        choices=["numpy", "tensorflow", "torch"],
        help="Optional output backend. Defaults to the same as input-backend.",
    )
    run_parser.add_argument(
        "--device",
        default="auto",
        help="Device policy for the pipeline adapter, e.g. auto, cpu, cuda, /GPU:0.",
    )
    run_parser.add_argument(
        "--shape-policy",
        default="strict",
        choices=["strict", "crop", "pad"],
        help="Shape policy to apply before calling the model.",
    )
    run_parser.add_argument(
        "--hr-multiple",
        type=int,
        default=1,
        help="Required HR divisibility multiple for the model.",
    )
    run_parser.add_argument(
        "--lr-multiple",
        type=int,
        default=1,
        help="Required LR divisibility multiple for the model.",
    )
    run_parser.add_argument(
        "--add-batch-dim",
        action="store_true",
        help="Add a batch dimension when converting image inputs for the pipeline.",
    )
    run_parser.add_argument(
        "--no-device-summary",
        action="store_true",
        help="Do not print the framework/device summary before running.",
    )

    return run_parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hyperbench",
        description="HyperBench command-line interface.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)

    return parser


def _run_command(args) -> int:
    if not args.no_device_summary:
        print_framework_device_summary()
        print()

    config = load_benchmark_config(args.config)
    adapter = _build_adapter_from_args(args)

    results = run_benchmark(adapter, config)

    success_count = sum(1 for row in results.rows if row.get("status") == "success")
    failure_count = sum(1 for row in results.rows if row.get("status") == "failed")

    print("Benchmark finished.")
    print("  rows:", len(results.rows))
    print("  success:", success_count)
    print("  failed:", failure_count)
    print("  csv:", Path(config.output_csv_path).resolve())

    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run_command(args)

        parser.print_help()
        return 1

    except HyperBenchError as exc:
        print("HyperBench error: {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print("Unexpected error: {}".format(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())