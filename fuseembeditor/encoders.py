"""Frozen encoder loading and attention-weighted pooling."""

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .config import (
    ATTENTION_POOL_EPS,
    CADUCEUS_KIND,
    CAUSAL_LM_KIND,
    DNABERT2_KIND,
    ENCODERS,
    HYENADNA_KIND,
    MASKED_LM_ENCODER_KINDS,
    MASKED_LM_KIND,
)


def load_encoder(model_name, device, cache_dir=None):
    """Load a frozen encoder and its tokeniser."""
    if model_name not in ENCODERS:
        raise KeyError(f"Unsupported encoder {model_name!r}; expected one of {list(ENCODERS)}")
    kind = ENCODERS[model_name]["kind"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if kind == CAUSAL_LM_KIND:
        # Only the LLM is large enough to warrant half precision on GPU.
        kwargs["torch_dtype"] = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(model_name, **kwargs)
    else:
        # The DNA encoders run their softmax in float32 and break under float16.
        kwargs["torch_dtype"] = torch.float32
        loader = AutoModelForMaskedLM if kind in MASKED_LM_ENCODER_KINDS else AutoModel
        model = loader.from_pretrained(model_name, **kwargs)

    if kind == DNABERT2_KIND:
        _disable_dnabert2_flash_attention(model)

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def _disable_dnabert2_flash_attention(model):
    """DNABERT-2 ships a Triton flash-attention kernel that no longer compiles.

    The module-level ``flash_attn_qkvpacked_func`` acts as the gate, so clearing it
    forces the bundled code down its plain PyTorch attention path.
    """
    import sys

    patched = False
    for name, module in list(sys.modules.items()):
        if "bert_layers" in name and hasattr(module, "flash_attn_qkvpacked_func"):
            module.flash_attn_qkvpacked_func = None
            patched = True
    if not patched:
        raise RuntimeError("Could not disable DNABERT-2 flash attention; its remote code layout changed.")


def encode_batch(model, kind, input_ids, attention_mask):
    """Return (token_embeddings, attentions) for one batch.

    ``attentions`` is None for the architectures that expose no attention matrices.
    """
    if kind == DNABERT2_KIND:
        # Custom forward returns (hidden_states, pooled) and rejects the usual kwargs.
        return model(input_ids=input_ids, attention_mask=attention_mask)[0], None

    if kind == HYENADNA_KIND:
        outputs = model(input_ids=input_ids)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return hidden, None

    if kind == CADUCEUS_KIND:
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1], None

    if kind == MASKED_LM_KIND:
        # NT registers only ForMaskedLM, so the representation comes from hidden_states.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, output_attentions=True)
        return outputs.hidden_states[-1], outputs.attentions

    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, output_attentions=True)
    return outputs.last_hidden_state, outputs.attentions


def attention_weighted_pool(
        token_embeddings, attention_mask, attentions=None, eps=ATTENTION_POOL_EPS):
    """Pool token representations into one vector per sample.

    With attention matrices available the weights are the model's own attention,
    averaged over layers, heads and query positions. Without them (Hyena, Mamba,
    DNABERT-2) the L2 norm of each token stands in as the importance proxy.
    """
    if attentions is None:
        weights = torch.norm(token_embeddings, dim=-1) * attention_mask.float()
    else:
        # Accumulate on CPU: NT emits a [batch, heads, seq, seq] tensor per layer.
        summed = None
        for layer_attention in attentions:
            layer_mean = layer_attention.mean(dim=1).cpu()
            summed = layer_mean if summed is None else summed + layer_mean
        averaged = (summed / len(attentions)).to(token_embeddings.device)
        weights = averaged.mean(dim=1) * attention_mask.float()

    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=eps)
    return (token_embeddings * weights.unsqueeze(-1)).sum(dim=1)
