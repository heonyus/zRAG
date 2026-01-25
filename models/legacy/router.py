"""
Document Selection 모듈
- CosineSelector: Cosine similarity 기반 (baseline)
- LearnedRouter: MLP 기반 학습 가능 router
- AttentionSelector: Cross-attention 기반 selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CosineSelector(nn.Module):
    """Cosine similarity 기반 document selection (non-parametric)"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_embed: torch.Tensor,
        doc_embeds: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embed: [batch, z_dim]
            doc_embeds: [num_docs, z_dim]
            k: top-k

        Returns:
            indices: [batch, k]
            scores: [batch, k]
        """
        # Normalize
        q_norm = F.normalize(query_embed, p=2, dim=-1)  # [batch, z_dim]
        d_norm = F.normalize(doc_embeds, p=2, dim=-1)   # [num_docs, z_dim]

        # Cosine similarity
        similarity = torch.matmul(q_norm, d_norm.t())  # [batch, num_docs]

        # Top-k
        top_k = torch.topk(similarity, k, dim=-1)
        return top_k.indices, top_k.values


class LearnedRouter(nn.Module):
    """MLP 기반 학습 가능 document router"""

    def __init__(self, z_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = z_dim // 2

        # Query transformation
        self.query_transform = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Document transformation
        self.doc_transform = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Temperature for score scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        query_embed: torch.Tensor,
        doc_embeds: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embed: [batch, z_dim]
            doc_embeds: [num_docs, z_dim]
            k: top-k

        Returns:
            indices: [batch, k]
            scores: [batch, k] (softmax-normalized)
        """
        batch_size = query_embed.size(0)
        num_docs = doc_embeds.size(0)

        # Transform
        q_transformed = self.query_transform(query_embed)  # [batch, hidden]
        d_transformed = self.doc_transform(doc_embeds)      # [num_docs, hidden]

        # Expand for pairwise scoring
        q_expanded = q_transformed.unsqueeze(1).expand(-1, num_docs, -1)  # [batch, num_docs, hidden]
        d_expanded = d_transformed.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_docs, hidden]

        # Concatenate and score
        combined = torch.cat([q_expanded, d_expanded], dim=-1)  # [batch, num_docs, hidden*2]
        raw_scores = self.score_head(combined).squeeze(-1)  # [batch, num_docs]

        # Temperature scaling
        scaled_scores = raw_scores / (self.temperature.abs() + 1e-8)

        # Top-k
        top_k_scores, top_k_indices = torch.topk(scaled_scores, k, dim=-1)

        # Softmax over selected (for gradient flow)
        top_k_probs = F.softmax(top_k_scores, dim=-1)

        return top_k_indices, top_k_probs

    def compute_full_scores(
        self,
        query_embed: torch.Tensor,
        doc_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """전체 documents에 대한 scores 반환 (retrieval loss 계산용)"""
        batch_size = query_embed.size(0)
        num_docs = doc_embeds.size(0)

        q_transformed = self.query_transform(query_embed)
        d_transformed = self.doc_transform(doc_embeds)

        q_expanded = q_transformed.unsqueeze(1).expand(-1, num_docs, -1)
        d_expanded = d_transformed.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([q_expanded, d_expanded], dim=-1)
        raw_scores = self.score_head(combined).squeeze(-1)

        scaled_scores = raw_scores / (self.temperature.abs() + 1e-8)
        return scaled_scores  # [batch, num_docs]


class AttentionSelector(nn.Module):
    """Cross-attention 기반 document selection"""

    def __init__(self, z_dim: int, num_heads: int = 4):
        super().__init__()
        self.z_dim = z_dim
        self.num_heads = num_heads
        self.head_dim = z_dim // num_heads

        assert z_dim % num_heads == 0, "z_dim must be divisible by num_heads"

        # Multi-head attention
        self.q_proj = nn.Linear(z_dim, z_dim)
        self.k_proj = nn.Linear(z_dim, z_dim)
        self.v_proj = nn.Linear(z_dim, z_dim)
        self.out_proj = nn.Linear(z_dim, 1)

        self.layer_norm = nn.LayerNorm(z_dim)
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))

    def forward(
        self,
        query_embed: torch.Tensor,
        doc_embeds: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embed: [batch, z_dim]
            doc_embeds: [num_docs, z_dim]
            k: top-k

        Returns:
            indices: [batch, k]
            scores: [batch, k]
        """
        batch_size = query_embed.size(0)
        num_docs = doc_embeds.size(0)

        # Query: [batch, 1, z_dim]
        q = self.q_proj(query_embed).unsqueeze(1)

        # Keys/Values: [1, num_docs, z_dim] → broadcast to [batch, num_docs, z_dim]
        doc_normed = self.layer_norm(doc_embeds)
        keys = self.k_proj(doc_normed).unsqueeze(0).expand(batch_size, -1, -1)
        values = self.v_proj(doc_normed).unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head reshape
        # q: [batch, 1, num_heads, head_dim] → [batch, num_heads, 1, head_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = keys.view(batch_size, num_docs, self.num_heads, self.head_dim).transpose(1, 2)
        v = values.view(batch_size, num_docs, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        # [batch, num_heads, 1, num_docs]

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.z_dim)

        # Score per document (average attention across heads)
        doc_scores = attn_scores.mean(dim=1).squeeze(1)  # [batch, num_docs]

        # Top-k
        top_k_scores, top_k_indices = torch.topk(doc_scores, k, dim=-1)
        top_k_probs = F.softmax(top_k_scores, dim=-1)

        return top_k_indices, top_k_probs
