#!/usr/bin/env python3
"""Run a paper benchmark model on a bundled dataset."""

import argparse
import json
from pathlib import Path

from benchmark.config import (
    BENCHMARK_MODELS,
    BENCHMARK_RUN_DIR,
    DATASETS,
    DEFAULT_BENCHMARK_DEVICE,
    DEFAULT_SEED,
    SMOKE_EPOCHS,
    SMOKE_SAMPLES,
)
from benchmark.train import run_benchmark


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=BENCHMARK_MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output_dir", type=Path, default=BENCHMARK_RUN_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_BENCHMARK_DEVICE)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_benchmark(
        args.model,
        args.dataset,
        args.output_dir,
        args.seed,
        device=args.device,
        epochs=SMOKE_EPOCHS if args.smoke else args.epochs,
        max_samples=SMOKE_SAMPLES if args.smoke else args.max_samples,
        save_checkpoint=args.save_checkpoint,
    )
    print(json.dumps(result["metrics"]["test"], indent=2))


if __name__ == "__main__":
    main()
