"""Regression head and the assembled FuseEmBEditor model."""

import torch.nn as nn

from .config import DROPOUT, HIDDEN_DIMS, LLM_REDUCED_DIM, PROJECTION_DIM
from .fusion import build_fusion


class RegressionHead(nn.Module):
    """MLP over the fused representation, ending in a sigmoid.

    The sigmoid constrains predictions to the [0, 1] efficiency range.
    """

    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class FusedRegressor(nn.Module):
    """Fusion module plus regression head.

    Batches arrive as the two modality embeddings concatenated along the feature
    axis; ``split_at`` is the DNA dimension used to separate them again.
    """

    def __init__(self, fusion_mod, reg_head, split_at):
        super().__init__()
        self.fusion_mod = fusion_mod
        self.reg_head = reg_head
        self.split_at = split_at

    def forward(self, x):
        fused = self.fusion_mod(x[:, :self.split_at], x[:, self.split_at:])
        return self.reg_head(fused)


def build_fused_model(dna_dim, llm_dim, fusion_strategy, llm_reduced_dim=LLM_REDUCED_DIM,
                      projection_dim=PROJECTION_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
    """Assemble the two-branch model."""
    fusion_mod, fused_dim = build_fusion(
        fusion_strategy, dna_dim, llm_dim, projection_dim, llm_reduced_dim)
    head = RegressionHead(fused_dim, hidden_dims, dropout)

    # Plain concatenation without equalisation has nothing to learn before the head,
    # so the model collapses to the head applied to the stacked embeddings.
    if next(fusion_mod.parameters(), None) is None:
        return head, fused_dim
    return FusedRegressor(fusion_mod, head, dna_dim), fused_dim


def build_solo_model(embedding_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
    """Single-modality baseline: a regression head on one frozen embedding."""
    return RegressionHead(embedding_dim, hidden_dims, dropout), embedding_dim
