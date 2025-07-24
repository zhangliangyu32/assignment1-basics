import torch
import torch.nn as nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=torch.sqrt(torch.tensor(2.0 / (in_features + out_features))))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... input_dim, output_dim input_dim -> ... output_dim')
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, mean=0.0, std=torch.sqrt(torch.tensor(2.0 / (num_embeddings + embedding_dim))))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.gain = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.gain)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Convert to float32 for numerical stability
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * (self.gain / norm)).to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear3 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        angles = einsum(torch.arange(max_seq_len, device=device).float(), inv_freq, 'seq_len, d_k_half -> seq_len d_k_half')
        cos_cached = angles.cos()
        sin_cached = angles.sin()
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions, :]
        sin = self.sin_cached[token_positions, :]
        x = rearrange(x, '... seq_len (d_model_half if_image) -> ... seq_len d_model_half if_image', if_image=2)
        x_real = x[..., 0]
        x_imag = x[..., 1]
        rotated_real = x_real * cos - x_imag * sin
        rotated_imag = x_real * sin + x_imag * cos
        return rearrange(torch.stack((rotated_real, rotated_imag), dim=-1), '... seq_len d_model_half if_image -> ... seq_len (d_model_half if_image)', if_image=2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x_stabilized = x - max_val

    exps = torch.exp(x_stabilized)

    sum_exps = torch.sum(exps, dim=dim, keepdim=True)

    return exps / sum_exps

def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = query.size(-1)
    scores = einsum(query, key, '... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k') / (d_k ** 0.5)

    if mask is not None:
        scores = torch.where(mask, scores, float(-torch.inf))

    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, value, '... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v')

class MultiheadSelfAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module, device=None, dtype=None):
        super(MultiheadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.rope = rope
        self.linear_q = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype)
        self.linear_k = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype)
        self.linear_v = Linear(d_model, self.d_v * num_heads, device=device, dtype=dtype)
        self.linear_out = Linear(self.d_v * num_heads, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        Q = rearrange(self.linear_q(x), 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k', num_heads=self.num_heads)
        K = rearrange(self.linear_k(x), 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k', num_heads=self.num_heads)
        V = rearrange(self.linear_v(x), 'batch_size seq_len (num_heads d_v) -> batch_size num_heads seq_len d_v', num_heads=self.num_heads) 
        if self.rope is not None:
            token_positions = rearrange(torch.arange(seq_len, device=x.device), 'seq_len -> 1 1 seq_len')
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        causal_mask = rearrange(causal_mask, 'seq_len_q seq_len_k -> 1 1 seq_len_q seq_len_k')
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        attn_output = rearrange(attn_output, 'batch_size num_heads seq_len d_v -> batch_size seq_len (num_heads d_v)')
        return self.linear_out(attn_output)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output
        ff_output = self.ff(self.norm2(x))
        return x + ff_output

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super(TransformerLM, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

