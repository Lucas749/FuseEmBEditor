# FuseEmBEditor

FuseEmBEditor predicts base-editing efficiency by combining frozen DNA-model embeddings with
frozen metadata-LLM embeddings. The repository includes the train, validation and test
splits used for the experiments.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install `.[ablations]` to use HyenaDNA or Caduceus-PH. DNABERT-2 and Caduceus-PH require their
Linux/CUDA dependencies.

## Generate embeddings

Generate both modalities for a dataset:

```bash
python scripts/generate_embeddings.py \
  --model InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
  --dataset BE4-2

python scripts/generate_embeddings.py \
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
python scripts/train.py --dataset BE4-2 --fusion projection
```

Runs are saved under `runs/<run-name>/` with the trained regression head, scalers, predictions,
configuration and metrics.

To reproduce every configuration, or one dataset only:

```bash
bash scripts/reproduce.sh
bash scripts/reproduce.sh BE4-2
```

## Data

The complete bundled datasets are `ALL_editors`, `ABE_combined`, `CBE_combined`, `ABE20m`,
`ABE8e`, `BE4-1`, `BE4-2` and `FNLS`. The DNA branch receives `grna`, `sequence` and
`pam_sequence`; all remaining model inputs are routed to the metadata branch.

## License

MIT, see [LICENSE](LICENSE).
