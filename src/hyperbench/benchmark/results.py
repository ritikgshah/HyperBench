# Author: Ritik Shah

"""Benchmark result containers for HyperBench."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class BenchmarkResults:
    """Container for benchmark result rows."""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add_row(self, row: Dict[str, Any]) -> None:
        self.rows.append(dict(row))

    def extend(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.add_row(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def to_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.rows:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([])
            return path

        fieldnames = self._collect_fieldnames()

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        return path

    def append_row_to_csv(
        self,
        path: Union[str, Path],
        row: Dict[str, Any],
        write_header_if_needed: bool = True,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = path.exists()
        fieldnames = list(row.keys())

        mode = "a" if file_exists else "w"
        with path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header_if_needed and not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return path

    def to_list(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.rows]

    def _collect_fieldnames(self) -> List[str]:
        seen = set()
        ordered = []

        preferred = [
            "scene_name",
            "shape_policy",
            "method_name",
            "psf_name",
            "sigma",
            "kernel_size",
            "downsampling_ratio",
            "msi_band_count",
            "spatial_snr",
            "spectral_snr",
            "fwhm_factor",
            "seed",
            "gt_shape",
            "lr_hsi_shape",
            "hr_msi_shape",
            "prediction_clipped",
            "prediction_clip_min",
            "prediction_clip_max",
            "num_parameters",
            "flops",
            "gpu_memory_mb",
            "gpu_peak_memory_mb",
            "model_inference_time_sec",
            "pipeline_train_time_sec",
            "RMSE",
            "PSNR",
            "SSIM",
            "UIQI",
            "ERGAS",
            "SAM",
            "status",
            "error",
            "runtime_seconds",
        ]

        for key in preferred:
            if any(key in row for row in self.rows):
                ordered.append(key)
                seen.add(key)

        for row in self.rows:
            for key in row.keys():
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)

        return ordered