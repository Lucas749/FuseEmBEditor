"""Dimension equalisation and multimodal fusion layers."""

import torch
import torch.nn as nn

from .config import (
    CONCATENATION_FUSION,
    FUSION_STRATEGIES,
    NO_DIM_REDUCTION,
    PROJECTION_DIM,
    PROJECTION_FUSION,
)


class ProjectionFusion(nn.Module):
    """Map both modalities into a shared space and add them.

    Additive fusion forces both branches to occupy the same coordinate system, so
    the regression head cannot partition the input and ignore one modality.
    """

    def __init__(self, dna_dim, llm_dim, proj_dim):
        super().__init__()
        self.dnabert_proj = nn.Sequential(
            nn.Linear(dna_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())

    def forward(self, dna_emb, llm_emb):
        return self.dnabert_proj(dna_emb) + self.llm_proj(llm_emb)


class GatedFusion(nn.Module):
    """Interpolate between the two projections with a learned per-dimension gate."""

    def __init__(self, dna_dim, llm_dim, proj_dim):
        super().__init__()
        self.dnabert_proj = nn.Linear(dna_dim, proj_dim)
        self.llm_proj = nn.Linear(llm_dim, proj_dim)
        self.gate = nn.Sequential(nn.Linear(dna_dim + llm_dim, proj_dim), nn.Sigmoid())

    def forward(self, dna_emb, llm_emb):
        gate = self.gate(torch.cat([dna_emb, llm_emb], dim=1))
        return gate * self.dnabert_proj(dna_emb) + (1 - gate) * self.llm_proj(llm_emb)


class ConcatFusion(nn.Module):
    """Stack the two vectors and leave the integration to the regression head."""

    def forward(self, dna_emb, llm_emb):
        return torch.cat([dna_emb, llm_emb], dim=1)


class DimReductionFusion(nn.Module):
    """Compress the metadata embedding to the DNA dimension before fusing.

    Without this the 4096-dim metadata branch contributes disproportionately more
    parameters and gradient signal than the 768-dim DNA branch.
    """

    def __init__(self, inner, llm_dim, reduced_dim):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Linear(llm_dim, reduced_dim), nn.LayerNorm(reduced_dim), nn.ReLU())
        self.inner = inner

    def forward(self, dna_emb, llm_emb):
        return self.inner(dna_emb, self.reducer(llm_emb))


def build_fusion(
        strategy, dna_dim, llm_dim, proj_dim=PROJECTION_DIM,
        llm_reduced_dim=NO_DIM_REDUCTION):
    """Return the fusion module and its output dimension."""
    if strategy not in FUSION_STRATEGIES:
        raise ValueError(f"Unknown fusion strategy {strategy!r}; expected one of {FUSION_STRATEGIES}")

    effective_llm_dim = llm_reduced_dim if llm_reduced_dim > 0 else llm_dim

    if strategy == CONCATENATION_FUSION:
        inner, out_dim = ConcatFusion(), dna_dim + effective_llm_dim
    elif strategy == PROJECTION_FUSION:
        inner, out_dim = ProjectionFusion(dna_dim, effective_llm_dim, proj_dim), proj_dim
    else:
        inner, out_dim = GatedFusion(dna_dim, effective_llm_dim, proj_dim), proj_dim

    if llm_reduced_dim > 0:
        return DimReductionFusion(inner, llm_dim, llm_reduced_dim), out_dim
    return inner, out_dim
