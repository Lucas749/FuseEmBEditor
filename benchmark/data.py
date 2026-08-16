"""Benchmark-specific preprocessing for the bundled dataset splits."""

import csv
import gzip

import numpy as np
from Bio.SeqUtils import MeltingTemp

from fuseembeditor.data import parse_fields, read_records, split_path

from .config import (
    ABERA_EDITOR,
    ABERA_FILTERS,
    BASE_EDITOR_KEY,
    BEDICT_BASES,
    BEDICT_V2,
    CONTEXT_LENGTH,
    CONTEXT_START,
    CRISPRON_BE,
    CRISPRON_SCORES_PATH,
    DATA_DIR,
    DEEPBASE_ABE_DATASETS,
    DEEPBASE_ABE_LENGTH,
    DEEPBASE_CBE_LENGTH,
    DEEPBASE_GRNA_START,
    DEEPBASEEDITOR,
    DEEPBE_PAM,
    DNA_BASES,
    FORECAST_BASES,
    FORECAST_BE,
    GRNA_KEY,
    GRNA_LENGTH,
    IGRNA_ABE,
    IGRNA_INPUT_SHAPE,
    LABEL_KEY,
    PAM_LENGTH,
    PAM_KEY,
    SEQUENCE_KEY,
    SPLITS,
    TEXT_KEY,
)


def one_hot(sequence, bases):
    mapping = {base: index for index, base in enumerate(bases)}
    encoded = np.zeros((len(sequence), len(bases)), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        if base in mapping:
            encoded[index, mapping[base]] = 1.0
    return encoded


def forecast_features(grna):
    features = [grna.count(base) for base in FORECAST_BASES]
    for base in grna:
        features.extend(float(base == candidate) for candidate in FORECAST_BASES)
    features.append(MeltingTemp.Tm_NN(grna))
    return np.asarray(features, dtype=np.float32)


def deepbase_sequence(fields, dataset):
    length = DEEPBASE_ABE_LENGTH if dataset in DEEPBASE_ABE_DATASETS else DEEPBASE_CBE_LENGTH
    start = DEEPBASE_GRNA_START - 1
    sequence = fields[SEQUENCE_KEY][start:start + length]
    if len(sequence) != length:
        return None
    return one_hot(sequence, DNA_BASES)[None, :, :]


def igrna_features(fields):
    from .models.igrna_abe.fuc import get_align

    grna = fields[GRNA_KEY]
    if len(grna) != GRNA_LENGTH or fields[SEQUENCE_KEY].find(grna) == -1:
        return None
    encoded = np.zeros(IGRNA_INPUT_SHAPE, dtype=np.float32)
    for index, features in enumerate(get_align(grna, grna)):
        encoded[0, :, index] = features
    return encoded


def bedict_sequence(fields):
    grna = fields[GRNA_KEY]
    sequence = fields[SEQUENCE_KEY]
    grna_start = sequence.find(grna)
    if len(grna) != GRNA_LENGTH or grna_start == -1:
        return None

    pam = fields[PAM_KEY]
    if len(pam) == 3:
        pam = sequence[
            grna_start + GRNA_LENGTH:grna_start + GRNA_LENGTH + PAM_LENGTH
        ]
    if len(pam) != PAM_LENGTH:
        return None
    return one_hot(grna + pam, BEDICT_BASES)


def deepbe_sequence(fields):
    grna = fields[GRNA_KEY]
    sequence = fields[SEQUENCE_KEY]
    grna_start = sequence.find(grna)
    if len(grna) != GRNA_LENGTH or grna_start == -1:
        return None

    start = max(0, grna_start - 4)
    end = min(len(sequence), grna_start + GRNA_LENGTH + len(fields[PAM_KEY]) + 3)
    context = sequence[start:end]
    if len(context) < CONTEXT_LENGTH:
        padding = "N" * (CONTEXT_LENGTH - len(context))
        context = padding + context if start == 0 else context + padding
    return one_hot(context[:CONTEXT_LENGTH], DNA_BASES)


def crispron_record(fields, score_cache):
    context = fields[SEQUENCE_KEY][CONTEXT_START:CONTEXT_START + CONTEXT_LENGTH]
    if len(context) != CONTEXT_LENGTH:
        return None
    off_score, on_score = score_cache[context]
    return context, off_score, on_score, fields[BASE_EDITOR_KEY]


def load_crispron_scores(path=CRISPRON_SCORES_PATH):
    scores = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            scores[row["seq30"]] = (
                float(row["CRISPRoff_score"]),
                float(row["CRISPRon"]),
            )
    return scores


def _prepare_record(model, dataset, fields, score_cache):
    if model == FORECAST_BE:
        grna = fields[GRNA_KEY]
        return forecast_features(grna) if len(grna) == GRNA_LENGTH else None
    if model == DEEPBASEEDITOR:
        return deepbase_sequence(fields, dataset)
    if model == IGRNA_ABE:
        return igrna_features(fields)
    if model == BEDICT_V2:
        return bedict_sequence(fields)
    if model == DEEPBE_PAM:
        return deepbe_sequence(fields)
    if model == CRISPRON_BE:
        return crispron_record(fields, score_cache)
    raise ValueError(f"Unknown benchmark model: {model}")


def _load_split(model, dataset, split, max_samples, score_cache):
    prepared = []
    labels = []
    for record in read_records(split_path(DATA_DIR, dataset, split)):
        fields = parse_fields(record[TEXT_KEY])
        if dataset in ABERA_FILTERS.get(model, ()) \
                and fields[BASE_EDITOR_KEY] == ABERA_EDITOR:
            continue
        model_input = _prepare_record(model, dataset, fields, score_cache)
        if model_input is None:
            continue
        prepared.append(model_input)
        labels.append(float(record[LABEL_KEY]))
        if max_samples and len(prepared) == max_samples:
            break
    if not prepared:
        raise ValueError(f"No valid records for {model} on {dataset}/{split}")
    return prepared, np.asarray(labels, dtype=np.float32)


def _crispron_inputs(rows, source_vocab):
    source_to_index = {source: index for index, source in enumerate(source_vocab)}
    sequences = np.stack([one_hot(row[0], DNA_BASES) for row in rows])
    off_scores = np.asarray([row[1] for row in rows], dtype=np.float32)[:, None]
    on_scores = np.asarray([row[2] for row in rows], dtype=np.float32)[:, None]
    sources = np.zeros((len(rows), len(source_vocab)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        sources[row_index, source_to_index[row[3]]] = 1.0
    return [sequences, off_scores, on_scores, sources]


def prepare_dataset(model, dataset, max_samples=None):
    score_cache = load_crispron_scores() if model == CRISPRON_BE else None
    raw_splits = {
        split: _load_split(model, dataset, split, max_samples, score_cache)
        for split in SPLITS
    }

    metadata = {"counts": {split: len(raw_splits[split][1]) for split in SPLITS}}
    if model == CRISPRON_BE:
        source_vocab = tuple(sorted({
            row[3] for split in SPLITS for row in raw_splits[split][0]
        }))
        metadata["source_vocab"] = source_vocab
        splits = {
            split: {
                "inputs": _crispron_inputs(raw_splits[split][0], source_vocab),
                "labels": raw_splits[split][1],
            }
            for split in SPLITS
        }
    else:
        splits = {
            split: {
                "inputs": np.stack(raw_splits[split][0]),
                "labels": raw_splits[split][1],
            }
            for split in SPLITS
        }
    return splits, metadata
