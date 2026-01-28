"""
Phase 1.5: Forward Wrapper for Evidence Generation

This module provides a dedicated forward wrapper for Phase 1.5 training.
It does NOT reuse WritePhaseModel.forward() directly, instead constructing
a custom forward pass for (z_i, query) -> evidence generation.

Key differences from Phase 1:
- Phase 1: z_i -> D_i (document regeneration)
- Phase 1.5: [z_i] + "Question: {query}\nEvidence:" -> evidence (extraction)

The wrapper ensures:
1. z_embed is computed with frozen z and projection
2. Loss is applied ONLY to evidence token positions
3. Attention mask correctly covers [z | prompt | evidence] segments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class Phase15ForwardWrapper:
    """
    Phase 1.5 전용 forward wrapper.

    WritePhaseModel의 LLM만 재사용하고, forward는 새로 구성:
    - Input: [z_embed] + [prompt_embed] + [evidence_embed[:-1]]
    - Loss: evidence 토큰 구간에만 적용
    """

    def __init__(self, base_model, tokenizer, device: str = "cuda"):
        """
        Args:
            base_model: WritePhaseModel instance (frozen z_to_embedding, alpha)
            tokenizer: LLM tokenizer
            device: Device string
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

        # References to base model components
        self.llm = base_model.llm
        self.z_to_embedding = base_model.z_to_embedding
        self.alpha = base_model.alpha
        self.m_tokens = base_model.m_tokens
        self.z_dim = base_model.z_dim
        self.hidden_size = base_model.hidden_size

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_z_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project z to LLM embedding space (FROZEN, no grad).

        Args:
            z: [batch, m_tokens, z_dim] or [m_tokens, z_dim]

        Returns:
            z_embed: [batch, m_tokens, hidden_size]
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)

        with torch.no_grad():
            # Clamp alpha to prevent scale collapse
            alpha_clamped = torch.clamp(self.alpha, min=0.5)

            # Project z to embedding space
            z_float = z.float()
            z_embed = alpha_clamped * self.z_to_embedding(z_float)

            # Convert to model dtype
            z_embed = z_embed.to(torch.bfloat16)

        return z_embed

    def forward_for_training(
        self,
        z: torch.Tensor,
        query_text: Union[str, List[str]],
        evidence_text: Union[str, List[str]],
        max_query_length: int = 128,
        max_evidence_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1.5 training forward pass.

        Input format: [z_embed] + [prompt_embed] + [evidence_embed[:-1]]
        Target: evidence_ids (full sequence, loss computed on evidence positions only)

        Args:
            z: [batch, m_tokens, z_dim] - frozen z vector
            query_text: str or List[str] - query strings
            evidence_text: str or List[str] - target evidence strings
            max_query_length: Maximum query tokens
            max_evidence_length: Maximum evidence tokens

        Returns:
            Dict with: loss, logits, z_positions, prompt_positions, evidence_positions
        """
        # Handle single sample
        if isinstance(query_text, str):
            query_text = [query_text]
        if isinstance(evidence_text, str):
            evidence_text = [evidence_text]

        batch_size = len(query_text)

        # Ensure z has batch dimension
        if z.dim() == 2:
            z = z.unsqueeze(0)
        if z.size(0) == 1 and batch_size > 1:
            z = z.expand(batch_size, -1, -1)

        # 1. Get z embedding (FROZEN)
        z_embed = self.get_z_embedding(z)  # [batch, m_tokens, hidden]

        # 2. Construct prompts: "Question: {query}\nEvidence:"
        prompts = [f"Question: {q}\nEvidence:" for q in query_text]

        # 3. Tokenize prompts
        prompt_encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_query_length,
            add_special_tokens=False,
        )
        prompt_ids = prompt_encoded["input_ids"].to(self.device)
        prompt_mask = prompt_encoded["attention_mask"].to(self.device)

        # 4. Tokenize evidence (target)
        evidence_encoded = self.tokenizer(
            evidence_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_evidence_length,
            add_special_tokens=False,
        )
        evidence_ids = evidence_encoded["input_ids"].to(self.device)
        evidence_mask = evidence_encoded["attention_mask"].to(self.device)

        # 5. Get embeddings
        embedding_layer = self.llm.get_input_embeddings()
        prompt_embed = embedding_layer(prompt_ids)  # [batch, prompt_len, hidden]
        evidence_embed = embedding_layer(evidence_ids)  # [batch, evidence_len, hidden]

        # 6. Concatenate: [z | prompt | evidence[:-1]] for teacher forcing
        # We predict evidence[1:] from evidence[:-1]
        combined_embed = torch.cat([
            z_embed,                      # [batch, m_tokens, hidden]
            prompt_embed,                 # [batch, prompt_len, hidden]
            evidence_embed[:, :-1, :],    # [batch, evidence_len-1, hidden]
        ], dim=1)

        # 7. Build attention mask
        z_mask = torch.ones(batch_size, self.m_tokens, device=self.device)
        combined_mask = torch.cat([
            z_mask,                       # [batch, m_tokens]
            prompt_mask,                  # [batch, prompt_len]
            evidence_mask[:, :-1],        # [batch, evidence_len-1]
        ], dim=1)

        # 8. Forward through LLM (LoRA is active here)
        outputs = self.llm(
            inputs_embeds=combined_embed,
            attention_mask=combined_mask,
            use_cache=False,
        )

        # 9. Compute loss ONLY on evidence positions
        # Positions: z (m_tokens) + prompt (prompt_len) -> evidence starts here
        m = self.m_tokens
        p = prompt_ids.size(1)  # prompt length (padded)
        e = evidence_ids.size(1)  # evidence length (padded)

        # Logits for predicting evidence tokens
        # At position (m + p + i - 1), we predict evidence[i]
        # So logits[:, m+p-1 : m+p+e-1, :] predict evidence[:e]
        start_pos = m + p - 1
        end_pos = start_pos + e

        if end_pos > outputs.logits.size(1):
            # Handle edge case where sequence is shorter
            end_pos = outputs.logits.size(1)
            e = end_pos - start_pos

        shift_logits = outputs.logits[:, start_pos:end_pos, :]  # [batch, e, vocab]
        shift_labels = evidence_ids[:, :e]  # [batch, e]

        # Mask out padding tokens in loss
        loss_mask = evidence_mask[:, :e]

        # Compute cross entropy loss
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view(batch_size, -1)

        # Apply mask and compute mean
        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return {
            "loss": loss,
            "logits": outputs.logits,
            "z_positions": (0, m),
            "prompt_positions": (m, m + p),
            "evidence_positions": (m + p - 1, end_pos),
        }

    @torch.no_grad()
    def generate_evidence(
        self,
        z: torch.Tensor,
        query_text: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate evidence from z and query.

        Args:
            z: [m_tokens, z_dim] or [1, m_tokens, z_dim]
            query_text: Query string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            top_p: Top-p sampling threshold

        Returns:
            Generated evidence string
        """
        # Ensure z has batch dimension
        if z.dim() == 2:
            z = z.unsqueeze(0)

        # 1. Get z embedding
        z_embed = self.get_z_embedding(z)

        # 2. Construct prompt
        prompt = f"Question: {query_text}\nEvidence:"
        prompt_encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_encoded["input_ids"].to(self.device)

        # 3. Get prompt embedding
        embedding_layer = self.llm.get_input_embeddings()
        prompt_embed = embedding_layer(prompt_ids)

        # 4. Concatenate [z | prompt]
        combined_embed = torch.cat([z_embed, prompt_embed], dim=1)

        # 5. Attention mask
        z_mask = torch.ones(1, self.m_tokens, device=self.device)
        combined_mask = torch.cat([
            z_mask,
            torch.ones_like(prompt_ids, dtype=torch.long),
        ], dim=1)

        # 6. Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embed,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 7. Decode (skip the input part)
        # The generate() with inputs_embeds returns the full sequence including embeddings
        # We need to skip the prefix length
        prefix_len = combined_embed.size(1)
        generated_ids = outputs[0, prefix_len:]

        evidence = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return evidence.strip()

    def get_embedding_stats(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics of z embedding for debugging.

        Args:
            z: [m_tokens, z_dim] or [1, m_tokens, z_dim]

        Returns:
            Dict with z and z_embed statistics
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)

        with torch.no_grad():
            alpha_clamped = torch.clamp(self.alpha, min=0.5)
            z_float = z.float()
            z_embed = alpha_clamped * self.z_to_embedding(z_float)

        return {
            "alpha": self.alpha.item(),
            "alpha_clamped": alpha_clamped.item(),
            "z_norm": z.norm().item(),
            "z_mean": z.mean().item(),
            "z_std": z.std().item(),
            "z_embed_norm": z_embed.norm().item(),
            "z_embed_mean": z_embed.mean().item(),
            "z_embed_std": z_embed.std().item(),
        }
