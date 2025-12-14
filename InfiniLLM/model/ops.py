# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Department of Electrical & Computer Engineering, University of Arizona, USA
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# ref: https://github.com/meta-llama/llama3/tree/main

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import torch.cuda.nvtx as nvtx
import time 

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
        

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_cache:bool = False, attn_method:str = "softmax", window_size:int = 10, delta:int = None):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.use_cache = use_cache
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # profiler
        self.time_record = []
        self.time_record_io = []
        self.token_count = 0 
        self.output_features = []

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )

        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        if (self.use_cache and attn_method not in ["linear", "swa_linear"]):
            # primary cache for linear attention
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )


            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )

            self.prompt_len = 0
            self._prompt_positions = None

        elif (self.use_cache and attn_method == "linear"):
            self.cache_s = torch.zeros(
                args.max_batch_size,
                self.n_local_heads,
                self.head_dim + 1,
                self.head_dim,
            )

            self.cache_z = torch.zeros(
                args.max_batch_size,
                self.n_local_heads,
                self.head_dim + 1,
            )
            self.prompt_len = 0
            self._prompt_positions = None

        elif (self.use_cache and attn_method == "swa_linear"):
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )

            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )

            self.cache_s = torch.zeros(
                args.max_batch_size,
                self.n_local_heads,
                self.head_dim + 1,
                self.head_dim,
            )

            self.cache_z = torch.zeros(
                args.max_batch_size,
                self.n_local_heads,
                self.head_dim + 1,
            )

            self.prompt_len = 0
            self._prompt_positions = None

        else:
            self.cache_k, self.cache_v = None, None
            self.prompt_len = 0
            self._prompt_positions = None

        self.last_attention_weights = None

    def __swa_forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            window_size: int = 128,
    ):
        """Sliding Window Attention with PROMPT RETENTION - prevents degradation"""
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        start_time_io = time.time()
        if self.use_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            if start_pos == 0:
                # Initialize on first call
                if seqlen > 1:
                    # For batch evaluation: cap prompt_len at window_size to enable true sliding window
                    # For generation: if sequence is short, use it as prompt
                    self.prompt_len = min(seqlen, window_size) if seqlen > window_size else seqlen
                else:
                    # Single token at start (autoregressive generation)
                    self.prompt_len = 0
                self._prompt_positions = torch.arange(0, self.prompt_len, device=xk.device) if self.prompt_len > 0 else torch.tensor([], dtype=torch.long, device=xk.device)
                self._window_positions = torch.arange(0, window_size, device=xk.device)

            current_len = start_pos + seqlen

            self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
            if current_len > self.prompt_len + window_size:
                window_start = current_len - window_size

                keys = torch.cat([
                    self.cache_k[:bsz, :self.prompt_len],
                    self.cache_k[:bsz, window_start:current_len]
                ], dim=1)

                values = torch.cat([
                    self.cache_v[:bsz, :self.prompt_len],
                    self.cache_v[:bsz, window_start:current_len]
                ], dim=1)

                self._cache_positions = torch.cat([
                    self._prompt_positions,
                    self._window_positions + window_start
                ])
            else:
                keys = self.cache_k[:bsz, :current_len]
                values = self.cache_v[:bsz, :current_len]
                self._cache_positions = torch.arange(0, current_len, device=xk.device)
        else:
            keys, values = xk, xv
            self._cache_positions = torch.arange(start_pos, start_pos + seqlen, device=xk.device)

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Synchronize and record IO time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time_io = time.time()
        io_elapsed_time = (end_time_io - start_time_io) * 1000  # Convert to milliseconds
        self.time_record_io.append(io_elapsed_time)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        overall_start_time = time.time()
        nvtx.range_push(f"sliding_window_attn_pos_{start_pos}")
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if hasattr(self, '_debug_enabled') and self._debug_enabled:
            if start_pos == 0 or start_pos % 10 == 0: 
                current_len_val = start_pos + seqlen
                print(f"[DEBUG Layer] Token {start_pos}: "
                      f"current_len={current_len_val}, "
                      f"prompt_len={self.prompt_len if self.use_cache else 'N/A'}, "
                      f"keys_shape={keys.shape}, "
                      f"scores_shape={scores.shape}, "
                      f"attending_to={keys.shape[2]} tokens")

        query_positions = torch.arange(start_pos, start_pos + seqlen, device=scores.device).unsqueeze(1)
        key_positions = self._cache_positions.unsqueeze(0)
        causal_mask = torch.where(query_positions >= key_positions, 0.0, float("-inf"))

        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        self.last_attention_weights = scores

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        nvtx.range_pop()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        overall_stop_time = time.time()

        elapsed_time = (overall_stop_time - overall_start_time) * 1000  # 
        self.time_record.append(elapsed_time)
        self.token_count += seqlen

        return self.wo(output)
    
    def __swa_linear_forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            window_size: int,
            mask: Optional[torch.Tensor],
            delta: Optional[int] = None
    ):
        
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        start_time_io = time.time()
        if self.use_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            if start_pos == 0:
                if seqlen > 1:
                    self.prompt_len = min(seqlen, window_size) if seqlen > window_size else seqlen
                else:
                    self.prompt_len = 0
                self._prompt_positions = torch.arange(0, self.prompt_len, device=xk.device) if self.prompt_len > 0 else torch.tensor([], dtype=torch.long, device=xk.device)
                self._window_positions = torch.arange(0, window_size, device=xk.device)

                if hasattr(self, 'cache_s_primary'):
                    self.cache_s_primary.zero_()
                    self.cache_z_primary.zero_()
                    self.cache_s.zero_()
                    self.cache_z.zero_()
                self._last_evicted_end = self.prompt_len

            current_len = start_pos + seqlen

            self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

            effective_delta = delta if delta is not None else 0
            if current_len > self.prompt_len + window_size:
                window_start = current_len - window_size
                linear_end = window_start + effective_delta

                keys = torch.cat([
                    self.cache_k[:bsz, :self.prompt_len],
                    self.cache_k[:bsz, linear_end:current_len]
                ], dim=1)

                values = torch.cat([
                    self.cache_v[:bsz, :self.prompt_len],
                    self.cache_v[:bsz, linear_end:current_len]
                ], dim=1)

                num_softmax_keys = self.prompt_len + (current_len - linear_end)
                self._cache_positions = torch.cat([
                    self._prompt_positions,
                    torch.arange(linear_end, current_len, device=xk.device)
                ])

                if not hasattr(self, '_last_evicted_end'):
                    self._last_evicted_end = self.prompt_len

                evicted_start = self._last_evicted_end
                evicted_end = window_start

                if evicted_end > evicted_start:
                    evicted_keys = self.cache_k[:bsz, evicted_start:evicted_end]
                    evicted_values = self.cache_v[:bsz, evicted_start:evicted_end]
                    self._last_evicted_end = evicted_end
                else:
                    evicted_keys = None
                    evicted_values = None

                w_delta_keys = self.cache_k[:bsz, window_start:linear_end]
                w_delta_values = self.cache_v[:bsz, window_start:linear_end]
            else:
                keys = self.cache_k[:bsz, :current_len]
                values = self.cache_v[:bsz, :current_len]
                self._cache_positions = torch.arange(0, current_len, device=xk.device)
                evicted_keys = None
                evicted_values = None
                w_delta_keys = None
                w_delta_values = None
        else:
            keys, values = xk, xv
            self._cache_positions = torch.arange(start_pos, start_pos + seqlen, device=xk.device)
            evicted_keys = None
            evicted_values = None
            w_delta_keys = None
            w_delta_values = None

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        if not hasattr(self, 'cache_s_primary'):
            self.cache_s_primary = torch.zeros_like(self.cache_s)
            self.cache_z_primary = torch.zeros_like(self.cache_z)

        self.cache_s_primary = self.cache_s_primary.to(xq)
        self.cache_z_primary = self.cache_z_primary.to(xq)
        self.cache_s = self.cache_s.to(xq)
        self.cache_z = self.cache_z.to(xq)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time_io = time.time()
        io_elapsed_time = (end_time_io - start_time_io) * 1000
        self.time_record_io.append(io_elapsed_time)

        overall_start_time = time.time()

        scale = 1.0 / math.sqrt(self.head_dim)

        if evicted_keys is not None:
            evicted_values_repeated = repeat_kv(evicted_values, self.n_rep)
            evicted_keys_repeated = repeat_kv(evicted_keys, self.n_rep)

            ones_k = torch.ones(bsz, evicted_keys.shape[1], self.n_local_heads, 1, device=evicted_keys.device, dtype=evicted_keys.dtype)
            phi_k = torch.cat([ones_k, evicted_keys_repeated], dim=-1)

            k_expanded = phi_k.unsqueeze(-1)
            v_expanded = evicted_values_repeated.unsqueeze(-2)
            self.cache_s_primary[:bsz] += (k_expanded * v_expanded).sum(dim=1)
            self.cache_z_primary[:bsz] += phi_k.sum(dim=1)

        # IMPORTANT: Reset cache_s and cache_z for the current window's w-delta region
        # These should NOT accumulate across windows, only cache_s_primary/cache_z_primary accumulate
        self.cache_s[:bsz].zero_()
        self.cache_z[:bsz].zero_()

        if w_delta_keys is not None and w_delta_keys.shape[1] > 0:
            w_delta_values_repeated = repeat_kv(w_delta_values, self.n_rep)
            w_delta_keys_repeated = repeat_kv(w_delta_keys, self.n_rep)

            ones_k = torch.ones(bsz, w_delta_keys.shape[1], self.n_local_heads, 1, device=w_delta_keys.device, dtype=w_delta_keys.dtype)
            phi_k = torch.cat([ones_k, w_delta_keys_repeated], dim=-1)

            k_expanded = phi_k.unsqueeze(-1)
            v_expanded = w_delta_values_repeated.unsqueeze(-2)
            self.cache_s[:bsz] = (k_expanded * v_expanded).sum(dim=1)
            self.cache_z[:bsz] = phi_k.sum(dim=1)

        q_i = xq

        S_total = self.cache_s_primary[:bsz] + self.cache_s[:bsz]
        K_sum_total = self.cache_z_primary[:bsz] + self.cache_z[:bsz]

        # Extract components from feature mapping φ(x) = [1, x]
        # S_total[:, :, 0, :] corresponds to V_sum (from constant 1)
        # S_total[:, :, 1:, :] corresponds to k⊗v term
        # K_sum_total[:, :, 0] corresponds to count (from constant 1)
        # K_sum_total[:, :, 1:] corresponds to K_sum (Σ k)

        V_sum = S_total[:, :, 0, :]  # [bsz, n_heads, head_dim]
        S = S_total[:, :, 1:, :]     # [bsz, n_heads, head_dim, head_dim]

        count = K_sum_total[:, :, 0:1]   # [bsz, n_heads, 1]
        K_sum = K_sum_total[:, :, 1:]    # [bsz, n_heads, head_dim]

        # Linear numerator: V_sum + (q_i/√d)·S
        V_sum_expanded = V_sum.unsqueeze(1)  # [bsz, 1, n_heads, head_dim]
        q_i_expanded = q_i.unsqueeze(-2)  # [bsz, seqlen, n_heads, 1, head_dim]
        S_expanded = S.unsqueeze(1)  # [bsz, 1, n_heads, head_dim, head_dim]
        q_S = torch.matmul(q_i_expanded, S_expanded).squeeze(-2) * scale  # [bsz, seqlen, n_heads, head_dim]

        linear_numerator = V_sum_expanded + q_S

        # Linear denominator: count + (q_i/√d)·K_sum
        count_expanded = count.unsqueeze(1)  # [bsz, 1, n_heads, 1]
        K_sum_expanded = K_sum.unsqueeze(1)  # [bsz, 1, n_heads, head_dim]
        q_K_sum = (q_i * K_sum_expanded).sum(dim=-1, keepdim=True) * scale  # [bsz, seqlen, n_heads, 1]

        linear_denominator = count_expanded + q_K_sum

        q_i_t = q_i.transpose(1, 2)
        k_j = keys.transpose(1, 2)
        v_j = values.transpose(1, 2)

        # Compute q_i k_j^T (unscaled for numerator)
        q_k_scores_unscaled = torch.matmul(q_i_t, k_j.transpose(2, 3))

        # Apply causal mask
        query_positions = torch.arange(start_pos, start_pos + seqlen, device=q_k_scores_unscaled.device).unsqueeze(1)
        key_positions = self._cache_positions.unsqueeze(0)
        causal_mask = torch.where(query_positions >= key_positions, 0.0, float("-inf"))

        q_k_scores_unscaled = q_k_scores_unscaled + causal_mask.unsqueeze(0).unsqueeze(0)

        # Standard scaled softmax attention for numerical stability
        # Both numerator and denominator use scaled scores
        q_k_scores_scaled = q_k_scores_unscaled * scale

        # Apply max trick for numerical stability
        max_scores = q_k_scores_scaled.max(dim=-1, keepdim=True)[0]
        q_k_scores_stable = q_k_scores_scaled - max_scores

        exp_scores = torch.exp(q_k_scores_stable.float()).type_as(q_i_t)

        # Softmax numerator: Σ exp((q_i k_j^T)/√d) v_j
        softmax_numerator = torch.matmul(exp_scores, v_j)
        softmax_numerator = softmax_numerator.transpose(1, 2).contiguous()

        # Softmax denominator: Σ exp((q_i k_j^T)/√d)
        softmax_denominator = exp_scores.sum(dim=-1, keepdim=True).transpose(1, 2)

        D = linear_denominator + softmax_denominator + 1e-6

        O_i = (linear_numerator + softmax_numerator) / D
        O_i = O_i.view(bsz, seqlen, -1)

        overall_stop_time = time.time()
        elapsed_time = (overall_stop_time - overall_start_time) * 1000
        self.time_record.append(elapsed_time)
        self.token_count += seqlen

        return self.wo(O_i)
    
    def __linear_d1_forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        scale = 1.0 / math.sqrt(self.head_dim)

        xq_scaled = xq * scale
        xk_scaled = xk * scale

        ones_q = torch.ones(bsz, seqlen, self.n_local_heads, 1, device=xq.device, dtype=xq.dtype)
        phi_q = torch.cat([ones_q, xq_scaled], dim=-1)

        ones_k = torch.ones(bsz, seqlen, self.n_local_kv_heads, 1, device=xk.device, dtype=xk.dtype)
        phi_k = torch.cat([ones_k, xk_scaled], dim=-1)

        phi_k_repeated = repeat_kv(phi_k, self.n_rep)
        xv_repeated = repeat_kv(xv, self.n_rep)

        start_time_io = time.time()
        if self.use_cache:
            self.cache_s = self.cache_s.to(xq)
            self.cache_z = self.cache_z.to(xq)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_io = time.time()
            io_elapsed_time = (end_time_io - start_time_io) * 1000
            self.time_record_io.append(io_elapsed_time)

            overall_start_time = time.time()
            for i in range(seqlen):
                q_i = phi_q[:, i, :, :]

                q_i_expanded = q_i.unsqueeze(-2)
                numerator = torch.matmul(q_i_expanded, self.cache_s[:bsz]).squeeze(-2)
                denominator = (q_i * self.cache_z[:bsz]).sum(dim=-1, keepdim=True) + 1e-6
                output_i = numerator / denominator


                k_i = phi_k_repeated[:, i, :, :]
                v_i = xv_repeated[:, i, :, :]

                # Compute outer product: k_i^T @ v_i for each head
                k_i_expanded = k_i.unsqueeze(-1)
                v_i_expanded = v_i.unsqueeze(-2)
                self.cache_s[:bsz] += k_i_expanded * v_i_expanded
                self.cache_z[:bsz] += k_i

                if i == 0:
                    output = output_i.unsqueeze(1)
                else:
                    output = torch.cat([output, output_i.unsqueeze(1)], dim=1)

            output = output.contiguous().view(bsz, seqlen, -1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            overall_stop_time = time.time()

            elapsed_time = (overall_stop_time - overall_start_time) * 1000
            self.time_record.append(elapsed_time)
            self.token_count += seqlen

        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_io = time.time()
            io_elapsed_time = (end_time_io - start_time_io) * 1000
            self.time_record_io.append(io_elapsed_time)

            overall_start_time = time.time()
            phi_q_t = phi_q.transpose(1, 2)
            phi_k_t = phi_k_repeated.transpose(1, 2)
            xv_t = xv_repeated.transpose(1, 2)

            kv = torch.matmul(phi_k_t.transpose(2, 3), xv_t)
            numerator = torch.matmul(phi_q_t, kv)

            k_sum = phi_k_t.sum(dim=2, keepdim=True)
            denominator = torch.matmul(phi_q_t, k_sum.transpose(2, 3)) + 1e-6
            output = numerator / denominator

            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            overall_stop_time = time.time()

            elapsed_time = (overall_stop_time - overall_start_time) * 1000
            self.time_record.append(elapsed_time)
            self.token_count += seqlen

        return self.wo(output) 


    def __vanilla_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
     
        start_time_io = time.time()
        if (self.use_cache):
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        else:
            keys, values = xk, xv

        keys = repeat_kv(
            keys, self.n_rep
        ) 
        values = repeat_kv(
            values, self.n_rep
        )  

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time_io = time.time()
        io_elapsed_time = (end_time_io - start_time_io) * 1000
        self.time_record_io.append(io_elapsed_time)

        xq = xq.transpose(1, 2)  
        keys = keys.transpose(1, 2)  
        values = values.transpose(
            1, 2
        ) 

        overall_start_time = time.time()
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask 
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        self.last_attention_weights = scores

        output = torch.matmul(scores, values)  
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        overall_stop_time = time.time()

        elapsed_time = (overall_stop_time - overall_start_time) * 1000
        self.time_record.append(elapsed_time)
        self.token_count += seqlen
        
        return self.wo(output)
    
    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            attn_method: str = "softmax",
            window_size: int = 128,
            delta: Optional[int] = None,
    ):
        """
        Forward pass that routes to appropriate attention mechanism
        Args:
            attn_method:
                - "softmax": standard full attention
                - "window": sliding window attention only
                - "linear": linear attention with Taylor degree-1 approximation
                - "swa_linear": hybrid sliding window + linear attention (evicted tokens go to linear memory)
            delta: For swa_linear only - number of recent tokens to attend with softmax (≤ window_size)
        """
        if attn_method == "window":
            return self.__swa_forward(x, start_pos, freqs_cis, mask, window_size)
        elif attn_method == "linear":
            return self.__linear_d1_forward(x, start_pos, freqs_cis, mask)
        elif attn_method == "swa_linear":
            return self.__swa_linear_forward(x, start_pos, freqs_cis, window_size, mask, delta)
        else:
            return self.__vanilla_forward(x, start_pos, freqs_cis, mask)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, use_cache:bool = False, attn_method:str = "softmax", window_size:int = 128, delta:int = None):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, use_cache, attn_method, window_size)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attn_method = attn_method
        self.window_size = window_size
        self.delta = delta

        self.hidden_state = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        atten_norms = self.attention_norm(x)

        # torch.cuda.nvtx.range_push("attention_computation")
        atten_out = self.attention(atten_norms, start_pos, freqs_cis, mask, self.attn_method, self.window_size, self.delta)
        # torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()

        h = x + atten_out
        self.hidden_state = atten_out
        out = h + self.feed_forward(self.ffn_norm(h))

        return out, self.hidden_state


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, use_cache:bool = False, attn_method = "softmax", window_size:int = 128, delta_values=None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if delta_values is not None:
            if isinstance(delta_values, int):
                delta_values = [delta_values] * params.n_layers
            elif isinstance(delta_values, (list, tuple)):
                assert len(delta_values) == params.n_layers, \
                    f"delta_values length ({len(delta_values)}) must match n_layers ({params.n_layers})"
            else:
                raise ValueError(f"delta_values must be int, list, or tuple, got {type(delta_values)}")

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            layer_delta = delta_values[layer_id] if delta_values is not None else None
            self.layers.append(TransformerBlock(layer_id, params, use_cache, attn_method, window_size, layer_delta))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        self.track_hidden_state = []

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        if len(self.track_hidden_state) > 0:
            self.track_hidden_state.clear()
            
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h, hidden_state = layer(h, start_pos, freqs_cis, mask)
            self.track_hidden_state.append(hidden_state)
        h = self.norm(h)
        output = self.output(h).float()
        return output, self.track_hidden_state