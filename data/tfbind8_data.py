"""Utilities to download and load TFBind-8 data without design-bench.

This script mirrors the TFBind-8 shard layout used by design-bench:
- tf_bind_8-<TRANSCRIPTION_FACTOR>/tf_bind_8-x-0.npy
- tf_bind_8-<TRANSCRIPTION_FACTOR>/tf_bind_8-y-0.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from huggingface_hub import snapshot_download

DEFAULT_REPO = "beckhamc/design_bench_data"
DEFAULT_TRANSCRIPTION_FACTOR = "SIX6_REF_R1"


def tfbind8_prefix(transcription_factor: str) -> str:
    return f"tf_bind_8-{transcription_factor}"


def download_tfbind8(
    transcription_factor: str = DEFAULT_TRANSCRIPTION_FACTOR,
    repo_id: str = DEFAULT_REPO,
    local_dir: str | os.PathLike[str] = "data/design_bench_data",
) -> Path:
    """Download the TFBind-8 x/y shards into a local folder.

    Returns the local dataset folder path:
    <local_dir>/tf_bind_8-<transcription_factor>
    """

    prefix = tfbind8_prefix(transcription_factor)
    allow_patterns = [
        f"{prefix}/tf_bind_8-x-0.npy",
        f"{prefix}/tf_bind_8-y-0.npy",
    ]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )

    dataset_dir = Path(local_dir, str(tfbind8_prefix(transcription_factor)))
    x_path = dataset_dir / "tf_bind_8-x-0.npy"
    y_path = dataset_dir / "tf_bind_8-y-0.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "TFBind-8 download did not produce expected files: "
            f"{x_path} and {y_path}"
        )

    return dataset_dir


def load_tfbind8(
    transcription_factor: str = DEFAULT_TRANSCRIPTION_FACTOR,
    local_dir: str | os.PathLike[str] = "data/design_bench_data",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load TFBind-8 into design-bench-compatible arrays.

    Returns:
    - x: integer array with shape [N, 8]
    - y: float array with shape [N, 1]
    """
    dataset_dir = Path(local_dir, str(tfbind8_prefix(transcription_factor)))
    x = np.load(dataset_dir / "tf_bind_8-x-0.npy")
    y = np.load(dataset_dir / "tf_bind_8-y-0.npy")

    # Normalize shape to the format design-bench expects.
    if y.ndim == 1:
        y = y[:, None]

    return x, y

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate TFBind-8 data without design-bench."
    )
    parser.add_argument(
        "--transcription-factor",
        default=DEFAULT_TRANSCRIPTION_FACTOR,
        help="TF name used in design-bench, e.g. SIX6_REF_R1",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("DB_HF_REPO", DEFAULT_REPO),
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--local-dir",
        default="data/design_bench_data",
        help="Local folder used to store downloaded dataset files",
    )
    args = parser.parse_args()

    dataset_dir = download_tfbind8(
        transcription_factor=args.transcription_factor,
        repo_id=args.repo_id,
        local_dir=args.local_dir,
    )
    x, y = load_tfbind8(
        transcription_factor=args.transcription_factor,
        local_dir=args.local_dir,
    )

    print(f"Saved to: {dataset_dir}")
    print(f"x shape={x.shape}, dtype={x.dtype}")
    print(f"y shape={y.shape}, dtype={y.dtype}")


if __name__ == "__main__":
    main()
