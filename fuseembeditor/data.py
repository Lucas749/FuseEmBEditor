"""Dataset loading and input segregation.

Each record in the BE-dataHIVE splits is a newline-separated ``key: value`` block.
The DNA branch receives only the nucleotide fields; the metadata branch receives
everything else, serialised as comma-separated key-value pairs.
"""

import gzip
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import (
    ALL_EDITORS_DATASET,
    BASE_METADATA_EXCLUDED_KEYS,
    COMBINED_DATASET_TOKEN,
    COMBINED_METADATA_EXCLUDED_KEYS,
    DNA_BRANCH,
    DNA_FIELD_ORDER,
    DNA_SEQUENCE_KEYS,
    LABEL_DECIMALS,
    SINGLE_EDITOR_METADATA_EXCLUDED_KEYS,
)


def split_path(data_dir, dataset, split):
    """Resolve a split file, accepting either the gzipped or plain form."""
    base = Path(data_dir) / f"{dataset}_{split}.jsonl"
    if base.exists():
        return base
    gz = base.with_suffix(".jsonl.gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"No split file for {dataset}/{split} under {data_dir}")


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def parse_fields(text):
    fields = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def dna_text(fields, concat=False):
    """gRNA, target and PAM in that order.

    ``concat`` drops the separators, which k-mer and single-nucleotide tokenisers
    need in order to read the sequence as one uninterrupted strand.
    """
    parts = [fields[k] for k in DNA_FIELD_ORDER if k in fields]
    return "".join(parts) if concat else " ".join(parts)


def metadata_text(fields, dataset):
    """Comma-separated key-value serialisation of the non-sequence fields.

    Categorical fields that are constant within a dataset carry no signal and are
    dropped: ``editor_type`` for the editor-type-combined sets, and editor/cell
    identity as well for the single-editor sets.
    """
    excluded = set(BASE_METADATA_EXCLUDED_KEYS | DNA_SEQUENCE_KEYS)
    if COMBINED_DATASET_TOKEN in dataset.lower():
        excluded |= COMBINED_METADATA_EXCLUDED_KEYS
    elif dataset != ALL_EDITORS_DATASET:
        excluded |= SINGLE_EDITOR_METADATA_EXCLUDED_KEYS

    return ", ".join(f"{k}:{v}" for k, v in fields.items() if k not in excluded)


class TextDataset(Dataset):
    """Tokenised view of one split, for a single branch.

    ``branch`` is ``"dna"`` or ``"metadata"`` and selects which serialisation is fed
    to the tokeniser.
    """

    def __init__(self, path, tokenizer, dataset, branch, max_length, concat_dna=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        for record in read_records(path):
            fields = parse_fields(record["text"])
            self.texts.append(dna_text(fields, concat_dna) if branch == DNA_BRANCH
                              else metadata_text(fields, dataset))
            self.labels.append(round(float(record["label"]), LABEL_DECIMALS))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        # HyenaDNA's tokeniser omits the attention mask.
        mask = encoding["attention_mask"].squeeze(0) if "attention_mask" in encoding \
            else torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class EmbeddingDataset(Dataset):
    """Precomputed embeddings paired with efficiency labels."""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}
