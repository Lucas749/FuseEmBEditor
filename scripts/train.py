#!/usr/bin/env python3
"""Train a fused or single-modality regressor on precomputed embeddings.

    python scripts/train.py --dataset BE4-2 --fusion projection
    python scripts/train.py --dataset BE4-2 --fusion projection --llm_reduced_dim 0
    python scripts/train.py --dataset BE4-2 --solo_model InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
    python scripts/train.py --dataset BE4-2 --solo_model Qwen/Qwen3-8B
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from fuseembeditor.config import (
    BATCH_SIZE,
    DATASETS,
    DEFAULT_FUSION,
    DEFAULT_SEED,
    DNA_ENCODERS,
    DROPOUT,
    EMBEDDING_DIR,
    EMBEDDING_FILE_TEMPLATE,
    EMBEDDING_POOLING_DIR,
    ENCODERS,
    EPOCHS,
    FUSION_STRATEGIES,
    HIDDEN_DIMS,
    LABEL_ALIGNMENT_ATOL,
    LABEL_DECIMALS,
    LEARNING_RATE,
    LLM_ENCODERS,
    LLM_REDUCED_DIM,
    NT_V2_250M,
    PROJECTION_DIM,
    QWEN3_8B,
    RUN_DIR,
    SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from fuseembeditor.data import EmbeddingDataset
from fuseembeditor.model import build_fused_model, build_solo_model
from fuseembeditor.training import evaluate_model, set_seed, standard_scale, train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--fusion", default=DEFAULT_FUSION, choices=FUSION_STRATEGIES)
    parser.add_argument("--dna_model", default=NT_V2_250M, choices=DNA_ENCODERS)
    parser.add_argument("--llm_model", default=QWEN3_8B, choices=LLM_ENCODERS)
    parser.add_argument("--solo_model", choices=list(ENCODERS),
                        help="Train a single-modality baseline on this encoder instead of fusing")
    parser.add_argument("--llm_reduced_dim", type=int, default=LLM_REDUCED_DIM,
                        help="Metadata dimension after equalisation; 0 disables it")
    parser.add_argument("--embedding_dir", type=Path, default=EMBEDDING_DIR)
    parser.add_argument("--output_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_embeddings(embedding_dir, model_name, dataset):
    """Load the three splits for one encoder."""
    base = Path(embedding_dir) / model_name.split("/")[-1] / dataset / EMBEDDING_POOLING_DIR
    splits = {}
    for split in SPLITS:
        path = base / EMBEDDING_FILE_TEMPLATE.format(split=split)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/generate_embeddings.py for {model_name} / {dataset} first.")
        data = torch.load(path, map_location="cpu")
        splits[split] = {
            "embeddings": data["embeddings"],
            "labels": torch.round(data["labels"], decimals=LABEL_DECIMALS),
        }
    return splits, splits[TRAIN_SPLIT]["embeddings"].shape[1]


def check_aligned(dna_splits, llm_splits):
    """The two branches must describe the same samples in the same order."""
    for split in SPLITS:
        n_dna = dna_splits[split]["embeddings"].shape[0]
        n_llm = llm_splits[split]["embeddings"].shape[0]
        if n_dna != n_llm:
            raise ValueError(f"Sample count mismatch in {split}: DNA={n_dna}, metadata={n_llm}")
        if not torch.allclose(
                dna_splits[split]["labels"], llm_splits[split]["labels"],
                atol=LABEL_ALIGNMENT_ATOL):
            raise ValueError(f"Label mismatch in {split} between the two embedding sets")


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.solo_model:
        run_name = f"solo_{args.solo_model.split('/')[-1]}_{args.dataset}"
    else:
        suffix = f"r{args.llm_reduced_dim}" if args.llm_reduced_dim > 0 else "no_eq"
        run_name = f"fused_{args.dna_model.split('/')[-1]}_{args.fusion}_{suffix}_{args.dataset}"

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}\nOutput: {output_dir}")

    scalers = {}
    if args.solo_model:
        splits, embedding_dim = load_embeddings(args.embedding_dir, args.solo_model, args.dataset)
        train_emb, val_emb, test_emb, scaler = standard_scale(
            splits[TRAIN_SPLIT]["embeddings"], splits[VAL_SPLIT]["embeddings"],
            splits[TEST_SPLIT]["embeddings"])
        scalers["embedding_scaler.pkl"] = scaler
        inputs = {TRAIN_SPLIT: train_emb, VAL_SPLIT: val_emb, TEST_SPLIT: test_emb}
        labels = {s: splits[s]["labels"] for s in SPLITS}
        model, fused_dim = build_solo_model(embedding_dim)
        setup = {"mode": "solo", "model": args.solo_model, "embedding_dim": embedding_dim}
    else:
        dna_splits, dna_dim = load_embeddings(args.embedding_dir, args.dna_model, args.dataset)
        llm_splits, llm_dim = load_embeddings(args.embedding_dir, args.llm_model, args.dataset)
        check_aligned(dna_splits, llm_splits)

        dna_train, dna_val, dna_test, dna_scaler = standard_scale(
            dna_splits[TRAIN_SPLIT]["embeddings"], dna_splits[VAL_SPLIT]["embeddings"],
            dna_splits[TEST_SPLIT]["embeddings"])
        llm_train, llm_val, llm_test, llm_scaler = standard_scale(
            llm_splits[TRAIN_SPLIT]["embeddings"], llm_splits[VAL_SPLIT]["embeddings"],
            llm_splits[TEST_SPLIT]["embeddings"])
        scalers["dna_embedding_scaler.pkl"] = dna_scaler
        scalers["llm_embedding_scaler.pkl"] = llm_scaler

        inputs = {
            TRAIN_SPLIT: torch.cat([dna_train, llm_train], dim=1),
            VAL_SPLIT: torch.cat([dna_val, llm_val], dim=1),
            TEST_SPLIT: torch.cat([dna_test, llm_test], dim=1),
        }
        labels = {s: dna_splits[s]["labels"] for s in SPLITS}
        model, fused_dim = build_fused_model(
            dna_dim, llm_dim, args.fusion, args.llm_reduced_dim)
        setup = {"mode": "fused", "dna_model": args.dna_model, "llm_model": args.llm_model,
                  "fusion_strategy": args.fusion, "dna_dim": dna_dim, "llm_dim": llm_dim,
                  "llm_reduced_dim": args.llm_reduced_dim, "projection_dim": PROJECTION_DIM}

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fused dimension: {fused_dim} | trainable parameters: {n_params:,}")

    loaders = {
        split: DataLoader(EmbeddingDataset(inputs[split], labels[split]),
                          batch_size=args.batch_size, shuffle=(split == TRAIN_SPLIT))
        for split in SPLITS
    }

    epoch_metrics = train_model(
        model, loaders[TRAIN_SPLIT], loaders[VAL_SPLIT], device, num_epochs=args.epochs)

    metrics, predictions = {}, {}
    for split in SPLITS:
        metrics[split], preds, actuals = evaluate_model(model, loaders[split], device, split)
        predictions[f"{split}_preds"] = preds
        predictions[f"{split}_actuals"] = actuals

    torch.save(model.state_dict(), output_dir / "regression_head.pth")
    for filename, scaler in scalers.items():
        joblib.dump(scaler, output_dir / filename)
    np.savez(output_dir / "predictions.npz", **predictions)

    results = {
        "run_name": run_name,
        "dataset": args.dataset,
        **setup,
        "fused_dim": fused_dim,
        "trainable_parameters": n_params,
        "seed": args.seed,
        "config": {"batch_size": args.batch_size, "num_train_epochs": args.epochs,
                   "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                   "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT},
        "train_metrics": metrics[TRAIN_SPLIT],
        "val_metrics": metrics[VAL_SPLIT],
        "test_metrics": metrics[TEST_SPLIT],
        "epoch_metrics": epoch_metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTest Pearson {metrics[TEST_SPLIT]['corr']:.4f} | "
          f"Spearman {metrics[TEST_SPLIT]['spearman']:.4f} | "
          f"R2 {metrics[TEST_SPLIT]['r2']:.4f} | "
          f"RMSE {metrics[TEST_SPLIT]['rmse']:.4f}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
