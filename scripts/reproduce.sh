#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python "$SCRIPT_DIR/reproduce.py" "$@"
#!/usr/bin/env bash
# Reproduce every run behind the paper: 40 embedding jobs, then 88 training runs.
#
#   bash scripts/reproduce.sh              # everything
#   bash scripts/reproduce.sh BE4-2        # one dataset
set -euo pipefail

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(ALL_editors ABE_combined CBE_combined ABE20m ABE8e BE4-1 BE4-2 FNLS)
fi

NT=InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
LLM=Qwen/Qwen3-8B
ABLATION_ENCODERS=(
    zhihan1996/DNABERT-2-117M
    LongSafari/hyenadna-medium-160k-seqlen-hf
    kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
)

for dataset in "${DATASETS[@]}"; do
    echo "=== $dataset: embedding ==="
    for encoder in "$NT" "$LLM" "${ABLATION_ENCODERS[@]}"; do
        python scripts/generate_embeddings.py --model "$encoder" --dataset "$dataset"
    done

    echo "=== $dataset: fusion strategies, with and without equalisation ==="
    for fusion in projection gated concatenation; do
        python scripts/train.py --dataset "$dataset" --fusion "$fusion"
        python scripts/train.py --dataset "$dataset" --fusion "$fusion" --llm_reduced_dim 0
    done

    echo "=== $dataset: DNA foundation model comparison ==="
    for encoder in "${ABLATION_ENCODERS[@]}"; do
        python scripts/train.py --dataset "$dataset" --fusion projection --dna_model "$encoder"
    done

    echo "=== $dataset: single-modality baselines ==="
    python scripts/train.py --dataset "$dataset" --solo_model "$NT"
    python scripts/train.py --dataset "$dataset" --solo_model "$LLM"
done
