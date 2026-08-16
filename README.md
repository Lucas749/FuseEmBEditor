# FuseEmBEditor

FuseEmBEditor predicts base-editing efficiency by combining frozen DNA-model embeddings with
frozen metadata-LLM embeddings. The repository includes the train, validation and test
splits used for the experiments.

## Install

```bash
git clone https://github.com/Lucas749/FuseEmBEditor.git
cd FuseEmBEditor
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.10–3.12 is supported. Install `.[ablations]` before full reproduction; Caduceus-PH
requires Linux and CUDA.

## Generate embeddings

Generate both modalities for a dataset:

```bash
fuseembeditor-embeddings \
  --model InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
  --dataset BE4-2

fuseembeditor-embeddings \
  --model Qwen/Qwen3-8B \
  --dataset BE4-2
```

The generated files are:

```text
embeddings/<model-name>/<dataset>/pooling_attention_weighted/train_embeddings.pt
embeddings/<model-name>/<dataset>/pooling_attention_weighted/val_embeddings.pt
embeddings/<model-name>/<dataset>/pooling_attention_weighted/test_embeddings.pt
```

## Train

```bash
fuseembeditor-train --dataset BE4-2 --fusion projection
```

Runs are saved under `runs/<run-name>/` with the trained regression head, scalers, predictions,
configuration and metrics.

To reproduce every configuration, or one dataset only:

```bash
pip install --no-build-isolation -e ".[ablations]"
fuseembeditor-reproduce
fuseembeditor-reproduce BE4-2
```

## Benchmarks

Install the benchmark dependencies, then run a full or one-epoch smoke benchmark:

```bash
pip install -e ".[benchmark]"
fuseembeditor-benchmark bedict_v2 --dataset BE4-2
fuseembeditor-benchmark bedict_v2 --dataset BE4-2 --smoke
```

Regenerate the bundled CRISPRon-BE scores with the original CRISPRoff 1.1.2 and
CRISPRon 1.0 code and models:

```bash
fuseembeditor-crispron-scores --output benchmark/data/crispron_scores.csv.gz
```

Copied upstream files under `benchmark/models` are unchanged. Package initializers and standalone
compatibility adapters under `benchmark/adapters` provide the integration layer:

- `forecast_be` — [FORECasT-BE](https://github.com/ananth-pallaseni/FORECasT-BE), original source and pretrained models
- `deepbaseeditor` — [DeepBaseEditor](https://github.com/MyungjaeSong/Paired-Library/tree/DeepCRISPR.info/DeepBaseEditor), TensorFlow compatibility adapter
- `igrna_abe` — [igRNA-ABE](https://github.com/shuaishuaigu/Igrna-abe), original feature code and model architecture
- `bedict_v2` — [BEDICT-V2](https://github.com/uzh-dqbm-cmi/BEDICT-V2), original `PredictionCNN`
- `deepbe_pam` — [DeepBE-PAM](https://github.com/NahyeKim/DeepBE), adapter generated from the original PAM architecture
- `crispron_be` — [CRISPRon-BE](https://github.com/RTH-tools/crispron-BE), [CRISPRoff](https://github.com/RTH-tools/crisproff) and [CRISPRon](https://github.com/RTH-tools/crispron), original scoring and training sources with standalone adapters

Third-party files retain the copyleft and Business Source License terms summarized in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
LLM-EmBEditor is available in the
[LLM-EmBEditor repository](https://github.com/Lucas749/LLM-EmBEditor).

## Data

The complete bundled datasets are `ALL_editors`, `ABE_combined`, `CBE_combined`, `ABE20m`,
`ABE8e`, `BE4-1`, `BE4-2` and `FNLS`. The DNA branch receives `grna`, `sequence` and
`pam_sequence`; all remaining model inputs are routed to the metadata branch.
The splits are derived from [BE-dataHIVE](https://doi.org/10.1186/s12859-024-05898-0)
under Apache License 2.0.

## License

FuseEmBEditor is MIT licensed; third-party benchmark files retain the terms listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
