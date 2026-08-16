"""Generate the CRISPRoff and CRISPRon scores used by CRISPRon-BE."""

import argparse
import csv
import gzip
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fuseembeditor.data import parse_fields, read_records, split_path

from ..config import (
    CONTEXT_LENGTH,
    CONTEXT_START,
    CRISPROFF_DIR,
    CRISPROFF_ENERGY_PATH,
    CRISPROFF_SCRIPT_PATH,
    CRISPRON_EVAL_PATH,
    CRISPRON_V0_MODELS_DIR,
    DATASETS,
    DATA_DIR,
    DEFAULT_CRISPRON_SCORES_OUTPUT,
    SEQUENCE_KEY,
    SPLITS,
    TEXT_KEY,
)


def collect_sequences(data_dir, datasets, max_sequences=None):
    sequences = set()
    for dataset in datasets:
        for split in SPLITS:
            for record in read_records(split_path(data_dir, dataset, split)):
                fields = parse_fields(record[TEXT_KEY])
                sequence = fields[SEQUENCE_KEY]
                context = sequence[CONTEXT_START:CONTEXT_START + CONTEXT_LENGTH]
                if len(context) != CONTEXT_LENGTH:
                    continue
                sequences.add(context)
                if max_sequences and len(sequences) == max_sequences:
                    return tuple(sorted(sequences))
    return tuple(sorted(sequences))


def _write_fastas(sequences, work_dir):
    fasta30 = work_dir / "30mers.fa"
    fasta23 = work_dir / "23mers.fa"
    identifiers = {}
    with open(fasta30, "w") as out30, open(fasta23, "w") as out23:
        for index, sequence in enumerate(sequences):
            identifier = f"s{index}"
            identifiers[identifier] = sequence
            out30.write(f">{identifier}\n{sequence}\n")
            out23.write(f">{identifier}\n{sequence[4:27]}\n")
    return fasta30, fasta23, identifiers


def _rnafold_environment(work_dir):
    environment = os.environ.copy()
    if shutil.which("RNAfold", path=environment.get("PATH")):
        return environment
    launcher = work_dir / "RNAfold"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "from benchmark.adapters.rnafold import main\n"
        "main()\n"
    )
    launcher.chmod(0o755)
    environment["PATH"] = f"{work_dir}{os.pathsep}{environment.get('PATH', '')}"
    return environment


def compute_scores(sequences):
    with tempfile.TemporaryDirectory(prefix="fuseembeditor-crispron-") as temporary:
        work_dir = Path(temporary)
        fasta30, fasta23, identifiers = _write_fastas(sequences, work_dir)
        crisproff_path = work_dir / "CRISPRparams.tsv"

        subprocess.run(
            [
                sys.executable,
                str(CRISPROFF_SCRIPT_PATH),
                "--guides",
                str(fasta23),
                "--specificity_report",
                str(work_dir / "CRISPRspec.tsv"),
                "--guide_params_out",
                str(crisproff_path),
                "--duplex_energy_params",
                str(CRISPROFF_ENERGY_PATH),
                "--no_azimuth",
            ],
            check=True,
            cwd=CRISPROFF_DIR,
            env=_rnafold_environment(work_dir),
        )

        model_paths = sorted(
            CRISPRON_V0_MODELS_DIR.glob("*.model.best"),
            key=lambda path: int(path.name.split(".")[0]),
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
        environment["TF_USE_LEGACY_KERAS"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(CRISPRON_EVAL_PATH),
                str(work_dir),
                str(fasta30),
                str(crisproff_path),
                *map(str, model_paths),
            ],
            check=True,
            env=environment,
        )

        with open(crisproff_path, newline="") as handle:
            off_scores = {
                row["guideID"]: float(row["CRISPRoff_score"])
                for row in csv.DictReader(handle, delimiter="\t")
            }
        with open(work_dir / "crispron.csv", newline="") as handle:
            on_scores = {
                row["ID"]: float(row["CRISPRon"])
                for row in csv.DictReader(handle)
            }
        return {
            sequence: (off_scores[identifier], on_scores[identifier])
            for identifier, sequence in identifiers.items()
        }


def write_scores(scores, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, newline="") as text:
                writer = csv.writer(text)
                writer.writerow(("seq30", "CRISPRoff_score", "CRISPRon"))
                writer.writerows((sequence, *scores[sequence]) for sequence in sorted(scores))


def generate_scores(
    output_path=DEFAULT_CRISPRON_SCORES_OUTPUT,
    data_dir=DATA_DIR,
    datasets=DATASETS,
    max_sequences=None,
):
    sequences = collect_sequences(data_dir, datasets, max_sequences=max_sequences)
    scores = compute_scores(sequences)
    write_scores(scores, output_path)
    return len(scores)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CRISPRON_SCORES_OUTPUT)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--max_sequences", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    count = generate_scores(
        output_path=args.output,
        data_dir=args.data_dir,
        datasets=args.datasets,
        max_sequences=args.max_sequences,
    )
    print(f"Wrote {count} sequences to {args.output}")


if __name__ == "__main__":
    main()
