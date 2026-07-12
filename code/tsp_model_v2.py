# tsp_model_v2.py — Enhanced POMO Actor with MatNet Edge Attention
#
# Upgrades over tsp_model.py:
#   1. MatNet-style edge attention: feeds the ACTUAL distance matrix into attention
#      (not just Euclidean distance). On OSRM, road distance ≠ Euclidean.
#   2. Backward-compatible: loads V6 checkpoints via strict=False
#   3. Variable n_nodes at inference (already supported, now explicit)
#
# Reference: Kwon et al., "Matrix Encoding Networks for Neural Combinatorial
#            Optimization" (NeurIPS 2021)

from __future__ import annotations
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached: Optional[int] = None
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _update_cos_sin(self, seq_len: int, device, dtype):
        if self._seq_len_cached is not None and self._seq_len_cached >= seq_len:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        self._seq_len_cached = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply_rotary(self, x, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, q, k):
        B, H, N, D = q.shape
        self._update_cos_sin(N, device=q.device, dtype=q.dtype)
        cos = self.cos_cached[:N]
        sin = self.sin_cached[:N]
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)


class MatNetMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with RoPE + dual geometric bias:
      - geo_mlp: bias from Euclidean distance (backward-compat with V6)
      - dist_mlp: bias from ACTUAL distance matrix (MatNet enhancement)

    When dist_matrix is None, only geo_mlp is used (same as V6).
    When dist_matrix is provided, both biases are summed (MatNet-enhanced).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        hidden = self.head_dim
        # V6-compatible: Euclidean distance bias
        self.geo_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, self.num_heads),
        )
        # MatNet + RRNCO-inspired asymmetric distance bias
        # Input: [forward_dist, backward_dist, asymmetry] = 3 features
        # This captures one-way streets, traffic direction, road topology
        # Ref: RRNCO "Neural Adaptive Bias" (ICLR 2026)
        self.dist_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, self.num_heads),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                dist_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, E = x.shape

        q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Euclidean distance bias (same as V6)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        eucl_dist = torch.norm(diff, dim=-1, keepdim=True)
        eucl_norm = eucl_dist / (eucl_dist.mean() + 1e-6)
        geo_bias = self.geo_mlp(eucl_norm).permute(0, 3, 1, 2)
        scores = scores + geo_bias

        # MatNet + RRNCO: asymmetric distance bias (when available)
        if dist_matrix is not None:
            forward_dist = dist_matrix.unsqueeze(-1)  # [B,N,N,1]  dist(i→j)
            backward_dist = dist_matrix.transpose(1, 2).unsqueeze(-1)  # dist(j→i)

            # Handle inf values
            finite_f = torch.isfinite(forward_dist)
            finite_b = torch.isfinite(backward_dist)
            safe_f = torch.where(finite_f, forward_dist, torch.zeros_like(forward_dist))
            safe_b = torch.where(finite_b, backward_dist, torch.zeros_like(backward_dist))

            mean_d = (safe_f.sum() + safe_b.sum()) / (finite_f.sum() + finite_b.sum() + 1e-6)
            f_norm = torch.where(finite_f, safe_f / (mean_d + 1e-6), torch.full_like(safe_f, 10.0))
            b_norm = torch.where(finite_b, safe_b / (mean_d + 1e-6), torch.full_like(safe_b, 10.0))
            asym = f_norm - b_norm  # captures one-way streets, directional bias

            dist_features = torch.cat([f_norm, b_norm, asym], dim=-1)  # [B,N,N,3]
            dist_bias = self.dist_mlp(dist_features).permute(0, 3, 1, 2)
            scores = scores + dist_bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, E)
        return self.out_proj(out)


class MatNetEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = 512,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.self_attn = MatNetMultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coords, dist_matrix=None):
        attn_out = self.self_attn(x, coords, dist_matrix)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class PointerDecoder(nn.Module):
    LOGIT_CLIP = 10.0  # standard in AM/POMO (Kool 2019, Kwon 2020)

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_emb: torch.Tensor, graph_ctx: torch.Tensor) -> torch.Tensor:
        B, N, E = node_emb.shape
        q = self.W_q(graph_ctx).unsqueeze(1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(node_emb).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = scores.mean(dim=1).squeeze(1)
        return self.LOGIT_CLIP * torch.tanh(logits)


class TSPActorV2(nn.Module):
    """
    Enhanced POMO actor with MatNet edge attention.

    Key difference from TSPActor:
      - Accepts optional dist_matrix [B,N,N] in forward()
      - Uses both Euclidean AND real distance for attention bias
      - Backward-compatible: works without dist_matrix (falls back to V6 behavior)

    Loading V6 checkpoints:
      model = TSPActorV2(...)
      load_v6_checkpoint(model, "checkpoint.pt")
    """

    def __init__(self, node_dim: int = 11, embed_dim: int = 128, num_heads: int = 8,
                 ff_dim: int = 512, num_layers: int = 6, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(node_dim, embed_dim)

        self.layers = nn.ModuleList([
            MatNetEncoderLayer(embed_dim, num_heads, ff_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.pointer_decoder = PointerDecoder(embed_dim, num_heads)

    def _encode(self, obs: torch.Tensor, n_nodes: int,
                dist_matrix: Optional[torch.Tensor] = None):
        B = obs.shape[0]
        node_dim = obs.shape[1] // n_nodes
        assert node_dim == self.node_dim

        x = obs.view(B, n_nodes, node_dim)
        coords = x[..., :2]
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, coords, dist_matrix)
        x = self.final_norm(x)
        return x, x.mean(dim=1)

    def forward(self, obs: torch.Tensor, n_nodes: int,
                dist_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        node_emb, graph_ctx = self._encode(obs, n_nodes, dist_matrix)
        return self.pointer_decoder(node_emb, graph_ctx)


def load_v6_checkpoint(model: TSPActorV2, path: str, device: str = "cuda"):
    """Load a V6 TSPActor checkpoint into TSPActorV2 (backward-compatible)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Map V6 key names to V2 key names
    mapped = {}
    for k, v in state.items():
        new_k = k
        # V6 uses RoPEMultiHeadAttention, V2 uses MatNetMultiHeadAttention
        # The weight names are identical for shared parameters
        mapped[new_k] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"[V2 LOAD] New parameters (randomly initialized): {len(missing)}")
        for m in missing:
            print(f"  + {m}")
    if unexpected:
        print(f"[V2 LOAD] Ignored V6 parameters: {len(unexpected)}")
    return model
