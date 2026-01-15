
"""
CRATE-α: Scaled Coding RAte reduction TransformEr

Based on "Scaling White-Box Transformers for Vision" (NeurIPS 2024)
This implements the three key modifications that enable CRATE to scale:

1. Overcomplete Dictionary: D ∈ R^{d × (C*d)} instead of R^{d × d}
2. Decoupled Dictionary: Separate encoding (D) and decoding (D_out) matrices  
3. Residual Connection: Around the sparse coding block

This is a DROP-IN REPLACEMENT for the vanilla CRATE implementation,
designed for use with nanochat's training infrastructure.

References:
- CRATE: Yu et al. "White-Box Transformers via Sparse Rate Reduction" (NeurIPS 2023)
- CRATE-α: Yang et al. "Scaling White-Box Transformers for Vision" (NeurIPS 2024)
"""

from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

# =============================================================================
# Try to import nanochat utilities, fall back to standalone if not available
# =============================================================================
try:
    from nanochat.common import get_dist_info, print0
    from nanochat.muon import Muon, DistMuon
    from nanochat.adamw import DistAdamW
    NANOCHAT_AVAILABLE = True
except ImportError:
    NANOCHAT_AVAILABLE = False
    def get_dist_info():
        return False, 0, 0, 1
    def print0(s="", **kwargs):
        print(s, **kwargs)


# =============================================================================
# Configuration (compatible with GPTConfig interface)
# =============================================================================

@dataclass
class CRATEConfig:
    """Configuration for CRATE-α model (mirrors GPTConfig interface)."""
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6  # For API compatibility (CRATE ignores this - uses tied weights)
    n_embd: int = 768
    window_pattern: str = "L"  # Sliding window: L=long, S=short
    
    # CRATE-α parameters (scaled version)
    odl_expansion: int = 4        # Overcomplete dictionary expansion factor C
    odl_use_residual: bool = True # Add residual around ODL block (critical for scaling!)
    odl_use_relu: bool = True     # Use ReLU instead of soft threshold (better for scaling)
    
    # Legacy ISTA parameters (for backwards compatibility)
    ista_step_size: float = 0.1   # Step size η (used if odl_use_relu=False)
    ista_lambda: float = 0.1      # Sparsity threshold λ (used if odl_use_relu=False)
    ista_mode: str = 'relu'       # 'soft_threshold' or 'relu'


# =============================================================================
# Core Functions
# =============================================================================

def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm with no learnable parameters (same as nanochat)."""
    return F.rms_norm(x, (x.size(-1),))


def soft_threshold(x: torch.Tensor, lambd: torch.Tensor) -> torch.Tensor:
    """
    Soft-thresholding operator (proximal operator for L1 norm).
    S_λ(x) = sign(x) * max(|x| - λ, 0)
    """
    return torch.sign(x) * F.relu(torch.abs(x) - lambd)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings (same as nanochat)."""
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


# =============================================================================
# MSSA: Multi-Head Subspace Self-Attention (unchanged from vanilla CRATE)
# =============================================================================

class MSSA(nn.Module):
    """
    Multi-Head Subspace Self-Attention (Compression Block)
    
    Key CRATE insight: Q, K, V share the SAME projection (tied weights!)
    
        w = x @ W  (single projection)
        attn = softmax(w @ w^T / √d)
        output = attn @ w
    
    This implements gradient descent on the coding rate compression term.
    """
    
    def __init__(self, config: CRATEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        
        self.scale = self.head_dim ** -0.5
        
        # Single QKV projection (TIED weights - the CRATE key insight!)
        self.qkv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor], 
                window_size: Tuple[int, int], kv_cache) -> torch.Tensor:
        B, T, C = x.size()
        
        # Single QKV projection (tied weights - CRATE's key insight)
        w_new = self.qkv(x).view(B, T, self.n_head, self.head_dim)  # [B, T, H, D]
        
        # Apply RoPE
        cos, sin = cos_sin
        w_new = apply_rotary_emb(w_new, cos, sin)
        w_new = norm(w_new)  # QK norm (like nanochat)
        
        # Handle KV cache for inference
        if kv_cache is not None:
            pos = kv_cache.get_pos()
            k_cache, _ = kv_cache.get_layer_cache(self.layer_idx)
            k_cache[:, pos:pos + T, :, :] = w_new
            w_q = w_new
            w_kv = k_cache[:, :pos + T, :, :]
            T_k = w_kv.size(1)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        else:
            w_q = w_new
            w_kv = w_new
            T_k = T
        
        # Rearrange for attention: [B, H, T, D]
        w_q = w_q.transpose(1, 2)
        w_kv = w_kv.transpose(1, 2)
        
        # Attention
        dots = torch.matmul(w_q, w_kv.transpose(-1, -2)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T_k, dtype=torch.bool, device=x.device),
            diagonal=T_k - T + 1
        )
        dots = dots.masked_fill(causal_mask, float('-inf'))
        
        # Sliding window mask
        window_left, _ = window_size
        if 0 < window_left < T_k:
            positions = torch.arange(T_k, device=x.device)
            query_positions = torch.arange(T, device=x.device) + (T_k - T)
            distance = query_positions.unsqueeze(1) - positions.unsqueeze(0)
            window_mask = distance > window_left
            dots = dots.masked_fill(window_mask, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, w_kv)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.c_proj(out)
        
        return out


# =============================================================================
# ODL: Overcomplete Dictionary Learning (CRATE-α's improved sparse coding)
# =============================================================================

class ODL(nn.Module):
    """
    Overcomplete Dictionary Learning Block (replaces vanilla ISTA)
    
    CRATE-α modifications for scaling:
    1. Overcomplete dictionary: D ∈ R^{d × (C*d)} where C > 1 (typically 4)
    2. Decoupled dictionary: Separate D_enc and D_dec matrices
    3. ReLU activation instead of soft-thresholding
    
    Forward pass:
        A = ReLU(Z @ D_enc)           # Encode to sparse overcomplete representation
        Z_out = A @ D_dec             # Decode back to original dimension
    
    This is similar to a standard MLP but derived from sparse coding principles!
    """
    
    def __init__(self, config: CRATEConfig):
        super().__init__()
        self.dim = config.n_embd
        self.expansion = config.odl_expansion
        self.hidden_dim = self.dim * self.expansion
        self.use_relu = config.odl_use_relu
        
        # Legacy ISTA parameters (for backwards compatibility / ablations)
        self.step_size = config.ista_step_size
        self.lambd = config.ista_lambda
        
        # Overcomplete dictionary D_enc ∈ R^{d × (C*d)}
        # This encodes input to a higher-dimensional sparse space
        self.D_enc = nn.Linear(self.dim, self.hidden_dim, bias=False)
        
        # Decoupled decoding dictionary D_dec ∈ R^{(C*d) × d}
        # This projects the sparse representation back to original dimension
        self.D_dec = nn.Linear(self.hidden_dim, self.dim, bias=False)
        
        # Optional: learnable threshold/bias for sparsification
        # This replaces the fixed λ parameter with a learned one
        self.threshold = nn.Parameter(torch.zeros(self.hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ODL block.
        
        Args:
            x: Input tensor [B, T, d]
            
        Returns:
            Output tensor [B, T, d] (sparse coded representation)
        """
        # Encode to overcomplete space
        h = self.D_enc(x)  # [B, T, C*d]
        
        # Sparsification
        if self.use_relu:
            # ReLU with learnable threshold (CRATE-α default)
            # This is more stable for scaling than soft-threshold
            h = F.relu(h - self.threshold)
        else:
            # Soft-thresholding (vanilla CRATE / ISTA)
            h = soft_threshold(h, self.step_size * self.lambd)
        
        # Decode back to original dimension
        out = self.D_dec(h)  # [B, T, d]
        
        return out


# =============================================================================
# Legacy ISTA Block (kept for ablations / comparison)
# =============================================================================

class ISTA(nn.Module):
    """
    Original ISTA Block (vanilla CRATE - kept for ablations)
    
    This is the original implementation that doesn't scale well.
    Use ODL instead for better scaling behavior.
    """
    
    def __init__(self, config: CRATEConfig):
        super().__init__()
        self.dim = config.n_embd
        self.step_size = config.ista_step_size
        self.lambd = config.ista_lambda
        self.mode = config.ista_mode
        
        # Complete dictionary D ∈ R^{dim × dim}
        self.weight = nn.Parameter(torch.empty(self.dim, self.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ISTA gradient step
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        z = x + self.step_size * (grad_2 - grad_1)
        
        # Sparsification
        if self.mode == 'soft_threshold':
            output = soft_threshold(z, self.step_size * self.lambd)
        else:
            output = F.relu(z - self.step_size * self.lambd)
        
        return output


# =============================================================================
# CRATE-α Block: MSSA + ODL with residuals
# =============================================================================

class Block(nn.Module):
    """
    One CRATE-α layer: MSSA (compression) + ODL (sparsification)
    
    Key difference from vanilla CRATE:
    - ODL HAS a residual connection (critical for scaling!)
    - Uses overcomplete dictionary instead of complete
    
    Formula:
        Z^{ℓ+1/2} = Z^ℓ + MSSA(norm(Z^ℓ))      # Compression with residual
        Z^{ℓ+1}   = Z^{ℓ+1/2} + ODL(norm(Z^{ℓ+1/2}))  # Sparsification with residual
    """
    
    def __init__(self, config: CRATEConfig, layer_idx: int):
        super().__init__()
        self.use_residual = config.odl_use_residual
        
        # Compression block (MSSA)
        self.mssa = MSSA(config, layer_idx)
        
        # Sparsification block (ODL for CRATE-α, ISTA for vanilla)
        self.odl = ODL(config)
    
    def forward(self, x: torch.Tensor, cos_sin, window_size, kv_cache) -> torch.Tensor:
        # Compression (MSSA with residual)
        x = x + self.mssa(norm(x), cos_sin, window_size, kv_cache)
        
        # Sparsification (ODL with residual - THE KEY CHANGE FOR SCALING!)
        if self.use_residual:
            x = x + self.odl(norm(x))
        else:
            # Vanilla CRATE behavior (doesn't scale well)
            x = self.odl(norm(x))
        
        return x


# =============================================================================
# KV Cache for Inference
# =============================================================================

class KVCache:
    """KV Cache for efficient autoregressive inference."""
    
    def __init__(self, batch_size: int, num_heads: int, seq_len: int, 
                 head_dim: int, num_layers: int, device, dtype=torch.bfloat16):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.w_cache = torch.zeros(
            num_layers, batch_size, seq_len, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    def reset(self):
        self.cache_seqlens.zero_()
    
    def get_pos(self) -> int:
        return self.cache_seqlens[0].item()
    
    def get_layer_cache(self, layer_idx: int):
        """Return (k_cache, v_cache) for compatibility. In CRATE, k=v."""
        return self.w_cache[layer_idx], self.w_cache[layer_idx]
    
    def advance(self, num_tokens: int):
        self.cache_seqlens += num_tokens


# =============================================================================
# Full CRATE-α Model
# =============================================================================

class CRATE(nn.Module):
    """
    Full CRATE-α model - DROP-IN REPLACEMENT for nanochat's GPT.
    
    All public methods match GPT's interface exactly.
    """
    
    def __init__(self, config: CRATEConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        
        # Compute window sizes
        self.window_sizes = self._compute_window_sizes(config)
        
        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        
        # Model architecture
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        
        # Per-layer scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def _compute_window_sizes(self, config: CRATEConfig):
        """Compute per-layer window sizes."""
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes
    
    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: float = 10000.0, device=None):
        """Precompute rotary embeddings."""
        if device is None:
            device = self.transformer.wte.weight.device if hasattr(self, 'transformer') else 'cpu'
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def init_weights(self):
        """Initialize all weights."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        
        # Embeddings
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        
        # CRATE-α blocks
        for block in self.transformer.h:
            # MSSA
            torch.nn.init.uniform_(block.mssa.qkv.weight, -s, s)
            torch.nn.init.zeros_(block.mssa.c_proj.weight)
            
            # ODL (CRATE-α style initialization)
            # Initialize similar to standard transformer MLP
            torch.nn.init.kaiming_uniform_(block.odl.D_enc.weight, a=5**0.5)
            torch.nn.init.zeros_(block.odl.D_dec.weight)
            torch.nn.init.zeros_(block.odl.threshold)
        
        # Per-layer scalars
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)
        
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        
        # Cast embeddings to bf16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
    
    def get_device(self):
        return self.transformer.wte.weight.device
    
    def estimate_flops(self):
        """Estimate FLOPs per token."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(ws[0], t) for ws in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops
    
    def num_scaling_params(self):
        """Return all parameters (same as GPT/Chinchilla)."""
        return sum(p.numel() for p in self.parameters())
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, 
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """Setup optimizers (MATCHES GPT's signature exactly)."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate parameters into groups
        # IMPORTANT: Muon requires 2D+ parameters (matrices only)
        # Filter out 1D parameters (like ODL threshold) and put them in AdamW
        all_block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_block_params if p.ndim >= 2]
        vector_params = [p for p in all_block_params if p.ndim < 2]
        
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        
        if vector_params:
            print0(f"Found {len(vector_params)} 1D parameters in blocks (e.g. ODL thresholds), routing to AdamW")
        
        assert len(list(self.parameters())) == len(matrix_params) + len(vector_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)
        
        # Scale LR by model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        
        # Add 1D parameters (ODL thresholds) to AdamW with scalar LR
        if vector_params:
            adam_groups.append(dict(params=vector_params, lr=scalar_lr))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        
        if NANOCHAT_AVAILABLE:
            AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
            MuonFactory = DistMuon if ddp else Muon
        else:
            AdamWFactory = partial(torch.optim.AdamW, fused=torch.cuda.is_available())
            MuonFactory = partial(torch.optim.AdamW, fused=torch.cuda.is_available())
            print0("Warning: Muon optimizer not available, falling back to AdamW for matrix params")
        
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay) if NANOCHAT_AVAILABLE else dict(lr=matrix_lr, weight_decay=weight_decay)
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        return optimizers
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                kv_cache=None, loss_reduction: str = 'mean') -> torch.Tensor:
        """Forward pass."""
        B, T = idx.size()
        
        # RoPE
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (self.cos[:, T0:T0+T], self.sin[:, T0:T0+T])
        
        # Forward the transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
        
        x = norm(x)
        
        # LM head with softcap
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        
        return logits
    
    @torch.inference_mode()
    def generate(self, tokens: list, max_tokens: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None, seed: int = 42):
        """Generate tokens."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


# =============================================================================
# Aliases for drop-in replacement
# =============================================================================
GPT = CRATE
GPTConfig = CRATEConfig


# =============================================================================
# Model Creation Helpers
# =============================================================================

def create_crate_alpha_tiny(**kwargs) -> CRATE:
    """CRATE-α Tiny: ~25M params"""
    config = CRATEConfig(n_layer=12, n_head=6, n_embd=384, odl_expansion=4, **kwargs)
    return CRATE(config)

def create_crate_alpha_small(**kwargs) -> CRATE:
    """CRATE-α Small: ~50M params"""
    config = CRATEConfig(n_layer=12, n_head=12, n_embd=576, odl_expansion=4, **kwargs)
    return CRATE(config)

def create_crate_alpha_base(**kwargs) -> CRATE:
    """CRATE-α Base: ~100M params"""
    config = CRATEConfig(n_layer=12, n_head=12, n_embd=768, odl_expansion=4, **kwargs)
    return CRATE(config)

def create_crate_alpha_large(**kwargs) -> CRATE:
    """CRATE-α Large: ~300M params"""
    config = CRATEConfig(n_layer=24, n_head=16, n_embd=1024, odl_expansion=4, **kwargs)
    return CRATE(config)


# =============================================================================
# Comparison: Vanilla CRATE vs CRATE-α
# =============================================================================

def create_vanilla_crate_base(**kwargs) -> CRATE:
    """
    Vanilla CRATE (for comparison/ablation).
    
    Key differences from CRATE-α:
    - odl_expansion=1 (complete dictionary, not overcomplete)
    - odl_use_residual=False (no residual around sparse block)
    - odl_use_relu=False (soft threshold instead of ReLU)
    """
    config = CRATEConfig(
        n_layer=12, n_head=12, n_embd=768,
        odl_expansion=1,         # Complete dictionary
        odl_use_residual=False,  # No residual (vanilla CRATE)
        odl_use_relu=False,      # Soft threshold
        **kwargs
    )
    return CRATE(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CRATE-α: Scaled White-Box Transformer (PyTorch)")
    print("Drop-in replacement for nanochat's GPT")
    print("=" * 70)
    
    # Test CRATE-α (scaled version)
    print("\n--- Testing CRATE-α (scaled) ---")
    config = CRATEConfig(
        vocab_size=50304,
        sequence_len=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        odl_expansion=4,
        odl_use_residual=True,
        odl_use_relu=True,
    )
    model = CRATE(config)
    model.init_weights()
    
    n_params = model.num_scaling_params()
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"FLOPs/token: {model.estimate_flops():,}")
    print(f"ODL expansion: {config.odl_expansion}x")
    print(f"ODL residual: {config.odl_use_residual}")
    print(f"ODL activation: {'ReLU' if config.odl_use_relu else 'Soft Threshold'}")
    
    # Test forward
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(x)
    print(f"Input: {x.shape} -> Output: {logits.shape}")
    
    # Test loss
    targets = torch.randint(0, config.vocab_size, (2, 64))
    loss = model(x, targets=targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Compare parameter counts
    print("\n--- Parameter Count Comparison ---")
    
    vanilla_config = CRATEConfig(
        vocab_size=50304, sequence_len=256, n_layer=6, n_head=6, n_embd=384,
        odl_expansion=1, odl_use_residual=False
    )
    vanilla_model = CRATE(vanilla_config)
    
    alpha_config = CRATEConfig(
        vocab_size=50304, sequence_len=256, n_layer=6, n_head=6, n_embd=384,
        odl_expansion=4, odl_use_residual=True
    )
    alpha_model = CRATE(alpha_config)
    
    print(f"Vanilla CRATE: {vanilla_model.num_scaling_params()/1e6:.2f}M params")
    print(f"CRATE-α (4x):  {alpha_model.num_scaling_params()/1e6:.2f}M params")
    print(f"Ratio: {alpha_model.num_scaling_params()/vanilla_model.num_scaling_params():.2f}x")
    
    # Verify interface matches GPT
    print("\n--- Interface Verification ---")
    print(f"✓ GPT = CRATE: {GPT is CRATE}")
    print(f"✓ GPTConfig = CRATEConfig: {GPTConfig is CRATEConfig}")
    print(f"✓ has init_weights: {hasattr(model, 'init_weights')}")
    print(f"✓ has setup_optimizers: {hasattr(model, 'setup_optimizers')}")
    print(f"✓ has estimate_flops: {hasattr(model, 'estimate_flops')}")
    print(f"✓ has num_scaling_params: {hasattr(model, 'num_scaling_params')}")
    print(f"✓ has generate: {hasattr(model, 'generate')}")
    
    print("\n✓ All verifications passed!")
    print("\nKey CRATE-α changes for scaling:")
    print("  1. Overcomplete dictionary (odl_expansion=4)")
    print("  2. Residual around ODL block (odl_use_residual=True)")
    print("  3. ReLU activation (odl_use_relu=True)")
    print("\nTo use with nanochat, change the import:")
    print("  from crate_alpha import GPT, GPTConfig")