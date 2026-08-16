"""Benchmark configuration."""

from pathlib import Path

from fuseembeditor.config import DATASETS, DATA_DIR, DEFAULT_SEED, RUN_DIR, SPLITS

FORECAST_BE = "forecast_be"
DEEPBASEEDITOR = "deepbaseeditor"
IGRNA_ABE = "igrna_abe"
BEDICT_V2 = "bedict_v2"
DEEPBE_PAM = "deepbe_pam"
CRISPRON_BE = "crispron_be"
LLM_EMBEDITOR = "llm_embeditor"

BENCHMARK_MODELS = (
    FORECAST_BE,
    DEEPBASEEDITOR,
    IGRNA_ABE,
    BEDICT_V2,
    DEEPBE_PAM,
    CRISPRON_BE,
)
MODEL_NAMES = {
    FORECAST_BE: "FORECasT-BE",
    DEEPBASEEDITOR: "DeepBaseEditor",
    IGRNA_ABE: "igRNA-ABE",
    BEDICT_V2: "BEDICT-V2",
    DEEPBE_PAM: "DeepBE-PAM",
    CRISPRON_BE: "CRISPRon-BE",
    LLM_EMBEDITOR: "LLM-EmBEditor",
}

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_RUN_DIR = RUN_DIR / "benchmark"
MODELS_DIR = BENCHMARK_DIR / "models"
CRISPRON_SCORES_PATH = BENCHMARK_DIR / "data" / "crispron_scores.csv.gz"
DEFAULT_CRISPRON_SCORES_OUTPUT = Path("crispron_scores.csv.gz")
CRISPRON_MODEL_DIR = MODELS_DIR / CRISPRON_BE
CRISPROFF_DIR = CRISPRON_MODEL_DIR / "crisproff-1.1.2"
CRISPROFF_SCRIPT_PATH = CRISPROFF_DIR / "CRISPRspec_CRISPRoff_pipeline.py"
CRISPROFF_ENERGY_PATH = CRISPROFF_DIR / "energy_dics.pkl"
CRISPRON_DIR = CRISPRON_MODEL_DIR / "crispron-1.0"
CRISPRON_EVAL_PATH = CRISPRON_DIR / "DeepCRISPRon_eval.py"
CRISPRON_V0_MODELS_DIR = CRISPRON_DIR / "CRISPRon_models_V0" / "best"
IGRNA_ARCH_PATH = MODELS_DIR / "igrna_abe" / "model_arch.json"
DEEPBE_ARCH_PATH = MODELS_DIR / "deepbe_pam" / "models_architecture.json"
DEEPBE_ARCH_MODEL = "PAM_variant_NRRH_model.h5"
UPSTREAM_REPOSITORIES = {
    FORECAST_BE: "https://github.com/ananth-pallaseni/FORECasT-BE",
    DEEPBASEEDITOR: "https://github.com/MyungjaeSong/Paired-Library/tree/DeepCRISPR.info/DeepBaseEditor",
    IGRNA_ABE: "https://github.com/shuaishuaigu/Igrna-abe",
    BEDICT_V2: "https://github.com/uzh-dqbm-cmi/BEDICT-V2",
    DEEPBE_PAM: "https://github.com/NahyeKim/DeepBE",
    CRISPRON_BE: "https://github.com/RTH-tools/crispron-BE",
    LLM_EMBEDITOR: "https://github.com/Lucas749/LLM-EmBEditor",
}

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = SPLITS
COMBINED_ABE_DATASET = "ABE_combined"
COMBINED_CBE_DATASET = "CBE_combined"
ALL_EDITORS_DATASET = "ALL_editors"
ABERA_EDITOR = "ABERA"
ABERA_FILTERS = {
    FORECAST_BE: frozenset((COMBINED_ABE_DATASET, ALL_EDITORS_DATASET)),
    DEEPBASEEDITOR: frozenset((COMBINED_ABE_DATASET,)),
}

GRNA_KEY = "grna"
SEQUENCE_KEY = "sequence"
PAM_KEY = "pam_sequence"
BASE_EDITOR_KEY = "base_editor"
LABEL_KEY = "label"
TEXT_KEY = "text"

DNA_BASES = ("A", "C", "G", "T")
FORECAST_BASES = ("G", "A", "C", "T")
BEDICT_BASES = ("A", "C", "T", "G")
GRNA_LENGTH = 20
DEEPBASE_GRNA_START = 10
DEEPBASE_ABE_LENGTH = 25
DEEPBASE_CBE_LENGTH = 24
CONTEXT_LENGTH = 30
CONTEXT_START = 6
PAM_LENGTH = 4
BEDICT_INPUT_LENGTH = GRNA_LENGTH + PAM_LENGTH

DEEPBASE_ABE_DATASETS = ("ABE20m", "ABE8e", COMBINED_ABE_DATASET, ALL_EDITORS_DATASET)

FORECAST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "loss": "squared_error",
    "criterion": "friedman_mse",
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "subsample": 1.0,
    "max_depth": 4,
    "random_state": None,
}
DEEPBASE_COMMON_PARAMS = {
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "dropout": 0.3,
    "kernel_size": 3,
    "patience": 10,
}
DEEPBASE_ABE_PARAMS = {
    "input_length": DEEPBASE_ABE_LENGTH,
    "conv_filters": 60,
    "hidden_dims": (80,),
}
DEEPBASE_CBE_PARAMS = {
    "input_length": DEEPBASE_CBE_LENGTH,
    "conv_filters": 150,
    "hidden_dims": (80, 60),
}
DEEPBASE_DATASET_OVERRIDES = {
    "BE4-2": {"epochs": 20},
    "FNLS": {"epochs": 20},
    COMBINED_CBE_DATASET: {
        "learning_rate": 1e-2,
        "patience": 25,
    },
}


def get_deepbase_params(dataset):
    params = dict(DEEPBASE_COMMON_PARAMS)
    editor_params = (
        DEEPBASE_ABE_PARAMS
        if dataset in DEEPBASE_ABE_DATASETS
        else DEEPBASE_CBE_PARAMS
    )
    params.update(editor_params)
    params.update(DEEPBASE_DATASET_OVERRIDES.get(dataset, {}))
    return params


IGRNA_PARAMS = {
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "conv_filters": 256,
    "conv_kernels": ((2, 13), (1, 13), (1, 13), (1, 13), (1, 13)),
    "hidden_dims": (256, 128, 32),
    "dropout": (0.25, 0.1, 0.1),
    "patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "min_learning_rate": 1e-6,
}
BEDICT_PARAMS = {
    "epochs": 300,
    "batch_size": 100,
    "learning_rate": 3e-4,
    "l2_reg": 0.01,
    "kernel_size": 2,
    "channels": (32, 64, 128),
    "hidden_dim": 64,
    "patience": 10,
    "scheduler_multiplier": 5,
}
DEEPBE_PARAMS = {
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "conv_filters": 2000,
    "kernel_size": 10,
    "hidden_dims": (1500, 100),
    "dropout": 0.3,
    "patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "min_learning_rate": 1e-6,
}
CRISPRON_PARAMS = {
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "conv_filters": (100, 70, 40),
    "conv_kernels": (3, 5, 7),
    "pool_size": 2,
    "hidden_dims": (80, 80, 60),
    "dropout": 0.3,
    "patience": 15,
    "scheduler_factor": 0.5,
    "scheduler_patience": 7,
    "min_learning_rate": 0.0,
}

IGRNA_INPUT_SHAPE = (1, 4, 23)
IGRNA_KERAS_LAYERS = ("InputLayer", "Conv2D", "Flatten", "Dense", "Dropout")
IGRNA_KERAS_INITIALIZERS = ("HeNormal", "Zeros", "GlorotUniform")

SMOKE_DATASET = "BE4-2"
SMOKE_EPOCHS = 1
SMOKE_SAMPLES = 10
DEFAULT_BENCHMARK_DEVICE = "cpu"

__all__ = [
    "BENCHMARK_MODELS",
    "BENCHMARK_RUN_DIR",
    "CRISPRON_SCORES_PATH",
    "DATASETS",
    "DATA_DIR",
    "DEFAULT_SEED",
    "MODEL_NAMES",
    "MODELS_DIR",
    "SPLITS",
    "UPSTREAM_REPOSITORIES",
]
