import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.randn((out_features, in_features), device=device, dtype=dtype), 
            std = 2/(in_features + out_features), 
            a = -3*2/(in_features + out_features), 
            b = 3*2/(in_features + out_features)
        ), requires_grad=True
        ) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight.T, x, "in_features out_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype), 
            requires_grad=True)
        
    def set(self, weights):
        self.weight = torch.nn.Parameter(weights.to(device=self.device, dtype=self.dtype))

    def get(self):
        return self.weight
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.randn(d_model, device=device, dtype=dtype), requires_grad=True)

    def set(self, weights):
        self.weight = weights.to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.linalg.vector_norm(x, dim=-1)**2 / self.d_model + self.eps)

        result = einsum(self.weight, x / torch.unsqueeze(rms,-1), "d_model, batch_size sequence_length d_model -> batch_size sequence_length d_model")
    
        return result.to(in_dtype)
    
def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        result = self.w1.forward(x)
        result = silu(result)
        w3_output = self.w3.forward(x)
        result.mul_(w3_output)
        final_output = self.w2.forward(result)
        del w3_output, result 
        return final_output
        return self.w2.forward(result * self.w3.forward(x))
    
class SiLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        result = self.w1.forward(x)
        result = silu(result)

        output = self.w2.forward(result)
        return output
        return self.w2.forward(result * self.w3.forward(x))

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.register_buffer('cos_sin_matrix', torch.randn(max_seq_len, d_k // 2, 2), persistent=False)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                self.cos_sin_matrix[i,k,0] = torch.cos(torch.tensor(i / (theta ** (2 * k / d_k))))
                self.cos_sin_matrix[i,k,1] = torch.sin(torch.tensor(i / (theta ** (2 * k / d_k))))
        self.cos_sin_matrix.to(device=device)
        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        x_temp = rearrange(x, "... (d_k_split split2) -> ... d_k_split split2", d_k_split=self.d_k // 2, split2=2)

        result = torch.stack((
            self.cos_sin_matrix[token_positions,:,0] * x_temp[..., 0] - self.cos_sin_matrix[token_positions,:,1] * x_temp[..., 1], 
            self.cos_sin_matrix[token_positions,:,1] * x_temp[..., 0] + self.cos_sin_matrix[token_positions,:,0] * x_temp[..., 1]
        ), dim=-1)
        result = result.view(x.shape)

        return result

def softmax(v, dim, temp=1.0):

    v /= temp

    v = torch.movedim(v, dim, 0)

    v = v - torch.amax(v, dim=0) 
    denom = torch.sum(torch.exp(v), 0)
    v_exp = torch.exp(v)
    out = v_exp / denom
    return torch.movedim(out, 0, dim)

def scaled_dot_product_attention(Q, K, V, mask):

    result = einsum(Q, K, "batch_size ... seq_len_1 d_k, batch_size ... seq_len_2 d_k -> batch_size ... seq_len_1 seq_len_2")
    result = result / np.sqrt(K.shape[-1])

    if mask is not None:
        result[...,~mask] = -torch.inf

    result = torch.nn.functional.softmax(result, dim=-1)
    result = einsum(result, V, "batch_size ... seq_len_1 seq_len_2, batch_size ... seq_len_2 d_v -> batch_size ... seq_len_1 d_v")

    return result


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, rope_theta=0, use_rope=False, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads 

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = False

        if rope_theta != 0:
            self.rope = ROPE(rope_theta, self.d_k, max_seq_len, device=device)
            self.use_rope = True

    def forward(self, x, token_positions=None):

        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).to(torch.bool)
        mask = ~mask

        q_x = self.q_proj.forward(x)
        k_x = self.k_proj.forward(x)
        v_x = self.v_proj.forward(x)

        # have to move sequence length to second to last dim 

        q_x = rearrange(q_x, "... (n_head d_k) -> ... n_head d_k", n_head=self.num_heads, d_k=self.d_k)
        k_x = rearrange(k_x, "... (n_head d_k) -> ... n_head d_k", n_head=self.num_heads, d_k=self.d_k)
        v_x = rearrange(v_x, "... (n_head d_k) -> ... n_head d_k", n_head=self.num_heads, d_k=self.d_v)

        q_x = rearrange(q_x, "... seq_len n_head d_k -> ... n_head seq_len d_k")
        k_x = rearrange(k_x, "... seq_len n_head d_k -> ... n_head seq_len d_k")
        v_x = rearrange(v_x, "... seq_len n_head d_k -> ... n_head seq_len d_k")

        if token_positions is None:
            token_positions = torch.arange(q_x.shape[-2])

        if self.use_rope:
            q_x = self.rope(q_x, token_positions)
            k_x = self.rope(k_x, token_positions)

        result = scaled_dot_product_attention(q_x, k_x, v_x, mask=mask)

        result = rearrange(result, "... n_head seq_len d_k -> ... seq_len (n_head d_k)")

        result = self.output_proj.forward(result)
        
        return result
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=2048, rope_theta=0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        # self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype)

    def forward(self, x):

        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        # y = self.ln1(x + self.attn(x))
        # z = self.ln2(y + self.ffn(y))
        # y = x + self.attn(x)
        # z = y + self.ffn(y)

        return z
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype))

        self.ln_final = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):

        x = self.token_embeddings(x)

        for i in range(self.num_layers):
            x = self.layers[i](x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x