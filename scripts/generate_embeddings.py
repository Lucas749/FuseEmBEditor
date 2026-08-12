#!/usr/bin/env python3
"""Precompute frozen embeddings for one encoder and one dataset.

Writes ``<output_dir>/<encoder>/<dataset>/pooling_attention_weighted/{split}_embeddings.pt``,
which is the layout ``train.py`` expects.

    python scripts/generate_embeddings.py --model InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --dataset BE4-2
    python scripts/generate_embeddings.py --model Qwen/Qwen3-8B --dataset BE4-2
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fuseembeditor.config import (
    DATASETS,
    DATA_DIR,
    EMBEDDING_DIR,
    EMBEDDING_FILE_TEMPLATE,
    EMBEDDING_POOLING_DIR,
    ENCODERS,
    SPLITS,
)
from fuseembeditor.data import TextDataset, split_path
from fuseembeditor.encoders import attention_weighted_pool, encode_batch, load_encoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(ENCODERS),
                        help="Encoder to run")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=EMBEDDING_DIR)
    parser.add_argument("--batch_size", type=int,
                        help="Defaults to the configured value for the encoder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dir", type=Path, help="Hugging Face cache directory")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = ENCODERS[args.model]
    device = torch.device(args.device)
    batch_size = args.batch_size or spec["batch_size"]

    model, tokenizer = load_encoder(args.model, device, args.cache_dir)

    out_dir = args.output_dir / args.model.split("/")[-1] / args.dataset / EMBEDDING_POOLING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        dataset = TextDataset(
            split_path(args.data_dir, args.dataset, split),
            tokenizer,
            args.dataset,
            branch=spec["branch"],
            max_length=spec["max_length"],
            concat_dna=spec.get("concat_dna", False),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"{split}: {len(dataset)} samples | example input: {dataset.texts[0][:120]}")

        embeddings, labels = [], []
        for batch in tqdm(loader, desc=f"Embedding {split}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                token_embeddings, attentions = encode_batch(
                    model, spec["kind"], input_ids, attention_mask)
                pooled = attention_weighted_pool(token_embeddings, attention_mask, attentions)
            embeddings.append(pooled.float().cpu())
            labels.append(batch["label"])

        torch.save({"embeddings": torch.cat(embeddings), "labels": torch.cat(labels)},
                   out_dir / EMBEDDING_FILE_TEMPLATE.format(split=split))

    print(f"Saved embeddings to {out_dir}")


if __name__ == "__main__":
    main()
