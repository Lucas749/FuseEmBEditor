"""FuseEmBEditor: modality-specialised embedding fusion for base editing efficiency."""

from .config import (
    DATASETS,
    DNA_ENCODERS,
    ENCODERS,
    FUSION_STRATEGIES,
    NT_V2_250M,
    QWEN3_8B,
    SPLITS,
)
from .data import EmbeddingDataset, TextDataset
from .fusion import build_fusion
from .model import RegressionHead, build_fused_model, build_solo_model
from .training import evaluate_model, set_seed, standard_scale, train_model

__all__ = [
    "DATASETS", "SPLITS", "EmbeddingDataset", "TextDataset",
    "DNA_ENCODERS", "ENCODERS", "NT_V2_250M", "QWEN3_8B",
    "FUSION_STRATEGIES", "build_fusion",
    "RegressionHead", "build_fused_model", "build_solo_model",
    "evaluate_model", "set_seed", "standard_scale", "train_model",
]
