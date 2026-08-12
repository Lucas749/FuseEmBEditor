#!/usr/bin/env python3
"""Run the complete embedding and training grid."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from fuseembeditor.config import (
    DATASETS,
    DEFAULT_FUSION,
    DNA_ENCODERS,
    ENCODERS,
    FUSION_STRATEGIES,
    NO_DIM_REDUCTION,
    NT_V2_250M,
    QWEN3_8B,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("datasets", nargs="*", metavar="DATASET")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown dataset: {', '.join(sorted(unknown))}")
    return args


def run(script, *args, dry_run=False):
    path = Path(__file__).with_name(script)
    command = [sys.executable, str(path), *args]
    if dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    datasets = args.datasets or DATASETS

    for dataset in datasets:
        for encoder in ENCODERS:
            run("generate_embeddings.py", "--model", encoder, "--dataset", dataset,
                dry_run=args.dry_run)

        for fusion in FUSION_STRATEGIES:
            run("train.py", "--dataset", dataset, "--fusion", fusion,
                dry_run=args.dry_run)
            run("train.py", "--dataset", dataset, "--fusion", fusion,
                "--llm_reduced_dim", str(NO_DIM_REDUCTION), dry_run=args.dry_run)

        for encoder in DNA_ENCODERS:
            if encoder != NT_V2_250M:
                run("train.py", "--dataset", dataset, "--fusion", DEFAULT_FUSION,
                    "--dna_model", encoder, dry_run=args.dry_run)

        run("train.py", "--dataset", dataset, "--solo_model", NT_V2_250M,
            dry_run=args.dry_run)
        run("train.py", "--dataset", dataset, "--solo_model", QWEN3_8B,
            dry_run=args.dry_run)


if __name__ == "__main__":
    main()
