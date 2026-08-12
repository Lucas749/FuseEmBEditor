"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
RUN_DIR = PROJECT_ROOT / "runs"

ALL_EDITORS_DATASET = "ALL_editors"
COMBINED_DATASET_TOKEN = "combined"
DATASETS = (
    ALL_EDITORS_DATASET,
    "ABE_combined",
    "CBE_combined",
    "ABE20m",
    "ABE8e",
    "BE4-1",
    "BE4-2",
    "FNLS",
)
SPLITS = ("train", "val", "test")
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = SPLITS

DNA_FIELD_ORDER = ("grna", "sequence", "pam_sequence")
DNA_SEQUENCE_KEYS = frozenset(DNA_FIELD_ORDER)
BASE_METADATA_EXCLUDED_KEYS = frozenset(("study_id", "efficiency_metric"))
COMBINED_METADATA_EXCLUDED_KEYS = frozenset(("editor_type",))
SINGLE_EDITOR_METADATA_EXCLUDED_KEYS = frozenset(("base_editor", "cell", "editor_type"))

QWEN3_8B = "Qwen/Qwen3-8B"
NT_V2_250M = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
DNABERT_2 = "zhihan1996/DNABERT-2-117M"
HYENADNA = "LongSafari/hyenadna-medium-160k-seqlen-hf"
CADUCEUS_PH = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

DNA_BRANCH = "dna"
METADATA_BRANCH = "metadata"
CAUSAL_LM_KIND = "causal_lm"
MASKED_LM_KIND = "masked_lm"
DNABERT2_KIND = "dnabert2"
HYENADNA_KIND = "hyenadna"
CADUCEUS_KIND = "caduceus"

ENCODERS = {
    QWEN3_8B: {
        "kind": CAUSAL_LM_KIND,
        "branch": METADATA_BRANCH,
        "dim": 4096,
        "max_length": 600,
        "batch_size": 4,
    },
    NT_V2_250M: {
        "kind": MASKED_LM_KIND,
        "branch": DNA_BRANCH,
        "dim": 768,
        "max_length": 512,
        "batch_size": 4,
        "concat_dna": True,
    },
    DNABERT_2: {
        "kind": DNABERT2_KIND,
        "branch": DNA_BRANCH,
        "dim": 768,
        "max_length": 512,
        "batch_size": 16,
        "concat_dna": False,
    },
    HYENADNA: {
        "kind": HYENADNA_KIND,
        "branch": DNA_BRANCH,
        "dim": 256,
        "max_length": 512,
        "batch_size": 32,
        "concat_dna": True,
    },
    CADUCEUS_PH: {
        "kind": CADUCEUS_KIND,
        "branch": DNA_BRANCH,
        "dim": 256,
        "max_length": 512,
        "batch_size": 32,
        "concat_dna": True,
    },
}
DNA_ENCODERS = (NT_V2_250M, DNABERT_2, HYENADNA, CADUCEUS_PH)
LLM_ENCODERS = (QWEN3_8B,)
MASKED_LM_ENCODER_KINDS = frozenset((MASKED_LM_KIND, CADUCEUS_KIND))

PROJECTION_FUSION = "projection"
GATED_FUSION = "gated"
CONCATENATION_FUSION = "concatenation"
FUSION_STRATEGIES = (PROJECTION_FUSION, GATED_FUSION, CONCATENATION_FUSION)
DEFAULT_FUSION = PROJECTION_FUSION
NO_DIM_REDUCTION = 0
HIDDEN_DIMS = (512, 256, 128)
PROJECTION_DIM = 512
LLM_REDUCED_DIM = 768
DROPOUT = 0.3

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 3
MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 42

LABEL_DECIMALS = 2
LABEL_ALIGNMENT_ATOL = 0.01
ATTENTION_POOL_EPS = 1e-9
EMBEDDING_POOLING_DIR = "pooling_attention_weighted"
EMBEDDING_FILE_TEMPLATE = "{split}_embeddings.pt"
