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
        ) # torch.empty instead of torch.randn

    def set(self, weight):
        self.weight = torch.nn.Parameter(weight.T.to(device=self.device, dtype=self.dtype))

    def get(self):
        return self.weight.T
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(self.in_features, self.out_features)
        # print(self.weights.data.shape)
        # print(x.shape)
        # print('-------------------------------------------')
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
        # print(self.embed[0].tolist())
        # print('-------------------------------------------')
        # print([self.embed[token_id].detach() for token_id in token_ids])
        # for token_id in token_ids:
        #     print(self.weight[token_id].tolist())
        # print(token_ids.shape)
        # print('blah')
        # print(self.weight[token_ids])
        # print(self.weight[token_ids].shape)
        return self.weight[token_ids]
        return torch.Tensor([self.weight[token_id].tolist() for token_id in token_ids])
        return 

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
        # print(rms)
        # print(rms.shape, x.shape)
        # print(self.weights / rms)

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

    def set(self, w1, w2, w3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x):
        # (self.w1.weight.shape, x.shape)
        result = self.w1.forward(x)
        result = silu(result)
        result = result * self.w3.forward(x)
        result = self.w2.forward(result)
        return result
    
class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.register_buffer('cos_sin_matrix', torch.randn(max_seq_len, d_k // 2, 2), persistent=False)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                # print(torch.tensor(i / (theta ** (2 * k / d_k))))
                # print(torch.cos(torch.Tensor(i / (theta ** (2 * k / d_k)))))
                self.cos_sin_matrix[i,k,0] = torch.cos(torch.tensor(i / (theta ** (2 * k / d_k))))
                self.cos_sin_matrix[i,k,1] = torch.sin(torch.tensor(i / (theta ** (2 * k / d_k))))
        self.cos_sin_matrix.to(device=device)
        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #print(token_positions)
        # print(x.shape)
        
        # reshape into d_k/2 2
        print(x.device)
        print(self.cos_sin_matrix.device)
        x_temp = rearrange(x, "... (d_k_split split2) -> ... d_k_split split2", d_k_split=self.d_k // 2, split2=2)
        # result = torch.zeros(x_temp.shape, device=x.device)
        print(x_temp.device)

        result = torch.stack((
            self.cos_sin_matrix[token_positions,:,0] * x_temp[..., 0] - self.cos_sin_matrix[token_positions,:,1] * x_temp[..., 1], 
            self.cos_sin_matrix[token_positions,:,1] * x_temp[..., 0] + self.cos_sin_matrix[token_positions,:,0] * x_temp[..., 1]
        ), dim=-1)
        # print(result.shape)
        result = result.view(x.shape)

        return result
        for i in token_positions:
            print(i)
            #print(x_temp.shape)
            #print(self.cos_sin_matrix.shape)
            #print(self.cos_sin_matrix[i,:,0].shape)
            new_append = torch.stack((
                self.cos_sin_matrix[i,:,0] * x_temp[..., 0] - self.cos_sin_matrix[i,:,1] * x_temp[..., 1], 
                self.cos_sin_matrix[i,:,1] * x_temp[..., 0] + self.cos_sin_matrix[i,:,0] * x_temp[..., 1]
            ),dim=-1)
            print(new_append.shape)
            result.append(torch.Tensor(new_append))

        # print(result)
        print(x.shape)
        result = torch.cat(result, dim=0)
        print(result.shape)

        result = torch.Tensor(result)
        result = torch.view(result, x.shape)

        return result
            

        # self.cos_sin_matrix[..., 0] * x[..., 0] - self.cos_sin_matrix[..., 1] * x[..., 1]
        # self.cos_sin_matrix[..., 1] * x[..., 0] + self.cos_sin_matrix[..., 0] * x[..., 1]
        # # x_temp = reshape(x, "... seq_len d_k -> seq_len d_k // 2 2")  ')
        # x_temp = rearrange(x, "... seq_len d_k -> ... d_k seq_len")
        # token_positions = torch.flatten(token_positions)
        # for i in range(len(token_positions)):
        #     # building R^i 
        #     R_i = torch.zeros(self.d_k, self.d_k)
        #     for k in range(self.d_k // 2):
        #         R_i[2 * k][2 * k] = self.cos_sin_matrix[i][k][0]
        #         R_i[2 * k + 1][2 * k] = self.cos_sin_matrix[i][k][1]
        #         R_i[2 * k][2 * k + 1] = -self.cos_sin_matrix[i][k][1]
        #         R_i[2 * k + 1][2 * k + 1] = self.cos_sin_matrix[i][k][0]
            
        # print(self.cos_sin_matrix.shape, token_positions.shape)
        # print(x.shape)
        # x = rearrange(x, 'batch_size seq_len d_model -> batch_size seq_len (d_model // 2) 2')
        # x = einsum(self.cos_sin_matrix[token_positions], x, "seq_len d_k_2 2, batch_size seq_len d_k_2 2 -> batch_size seq_len d_k_2 2")
        # x = rearrange(x, 'batch_size seq_len d_k_2 2 -> batch_size seq_len d_model')
        return x

def softmax(v, dim, temp=1.0):

    v /= temp

    v = torch.movedim(v, dim, 0)
    v_max = torch.amax(v, dim=0)
    v = v - v_max 
    denom = torch.sum(torch.exp(v), 0)
    # print(f"Before exp: v.requires_grad={v.requires_grad}, v.grad={v.grad_fn}")
    # v = torch.exp(v)
    v_exp = torch.exp(v)
    # print(f"After exp: v_exp.requires_grad={v_exp.requires_grad}, v_exp.grad={v_exp.grad_fn}")
    out = v_exp/ denom
    return torch.movedim(out, 0, dim)

def scaled_dot_product_attention(Q, K, V, mask):

    result = einsum(Q, K, "batch_size ... seq_len_1 d_k, batch_size ... seq_len_2 d_k -> batch_size ... seq_len_1 seq_len_2")
    result = result / np.sqrt(K.shape[-1])

    # print(torch.inf * (~mask).long())
    # print(result)

    if mask is not None:
        # print(mask.shape)
        # print(result.shape)
        result[...,~mask] = -torch.inf
        # result[~mask] = -torch.inf

    # print(result)

    # print(f"Before sdpa: result.requires_grad={result.requires_grad}, result.grad={result.grad_fn}")

    result = softmax(result, dim=-1)
    result = einsum(result, V, "batch_size ... seq_len_1 seq_len_2, batch_size ... seq_len_2 d_v -> batch_size ... seq_len_1 d_v")
    # print(f"after sdpa: result.requires_grad={result.requires_grad}, result.grad={result.grad_fn}")

    return result


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, rope_theta=0, use_rope=True, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads 

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            # print(rope_theta)
            self.rope = ROPE(rope_theta, self.d_k, max_seq_len)
        self.use_rope = use_rope

    def forward(self, x, token_positions=None):

        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).to(torch.bool)
        mask = ~mask
        # print(mask)

        q_x = self.q_proj.forward(x)
        k_x = self.k_proj.forward(x)
        v_x = self.v_proj.forward(x)

        # x_temp = rearrange(x, "... (d_model_split split) -> ... d_model_split split", d_model_split=self.num_heads, split=self.d_k)
        # q_temp = rearrange(self.q_proj, "(d_model_split split) d_model -> d_model_split (split d_model)", d_model_split=self.d_k, split=self.num_heads)

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
        # print(f"after mhsa: result.requires_grad={result.requires_grad}, result.grad={result.grad_fn}")
        
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
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype)

    def forward(self, x):

        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        # print(f"after block: result.requires_grad={z.requires_grad}, result.grad={z.grad_fn}")

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
        # print(f"after lm: result.requires_grad={x.requires_grad}, result.grad={x.grad_fn}")

        return x