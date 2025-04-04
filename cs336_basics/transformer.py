import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps, weights):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = weights

    def forward(self, x):
        rms = torch.linalg.vector_norm(x, dim=-1)**2 / self.d_model + self.eps 
        rms = torch.sqrt(rms)
        x = torch.movedim(x, -1, 0)
        normed = x / rms
        return torch.movedim(normed, 0, -1) * self.weights
    
class GELU(nn.Module):
    def forward(self, x):
        return x * (1 + torch.erf(x/np.sqrt(2))) / 2
    
class FFN(nn.Module):
    def __init__(self, weights1, weights2):
        super().__init__()
        self.weights1 = weights1
        self.weights2 = weights2 

    def forward(self, x):
        w1 = self.weights1
        w2 = self.weights2

        x = torch.unsqueeze(x, -1)
        x = w1 @ x

        gelu = GELU()
        x = gelu(x)

        x = torch.squeeze(w2 @ x)
        return x
    
def softmax(v, dim):

    v = torch.movedim(v, dim, 0)
    v_max = torch.amax(v, dim=0)
    v = v - v_max 
    denom = torch.sum(torch.exp(v), 0)
    v = torch.exp(v)
    v /= denom
    return torch.movedim(v, 0, dim)

def scaled_dot_product_attention(K, Q, V, mask):
    n = Q.shape[-2]
    m = K.shape[-2]
    d_k = K.shape[-1]
    d_v = V.shape[-1]

    # K = K.view(-1, m, d_k)
    # Q = Q.view(-1, n, d_k)
    # V = V.view(-1, m, d_v)

    attn = torch.bmm(Q, K.permute(0, 2, 1)) 
    attn = attn / np.sqrt(d_k)

    attn = softmax(attn, -1)

    mask = ~mask
    
    attn = attn * mask.to(torch.float32) 


    attn = attn / attn.sum(dim=-1, keepdim=True)

    attn = torch.bmm(attn, V)

    return attn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop

    def forward(self, k, q, v, out_proj, x):

        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).to(torch.bool)

        multihead = scaled_dot_product_attention(x@(k.T), x@(q.T), x@(v.T), mask)

        multihead = multihead @ out_proj

        return multihead
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, ln1, ln2, ffn1, ffn2, k_proj, q_proj, v_proj, out_proj):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop

        self.ln1 = ln1 
        self.ln2 = ln2 
        self.ffn1 = ffn1 
        self.ffn2 = ffn2 
        self.k_proj = k_proj 
        self.q_proj = q_proj 
        self.v_proj = v_proj
        self.out_proj = out_proj

    def forward(self, x):

        RMSNorm1 = RMSNorm(self.d_model, 1e-5, self.ln1)
        RMSNorm2 = RMSNorm(self.d_model, 1e-5, self.ln2)
        FF = FFN(self.ffn1, self.ffn2)
        MHSA = MultiheadSelfAttention(self.d_model, self.num_heads, self.attn_pdrop)

        y = x + MHSA(self.k_proj, self.q_proj, self.v_proj, self.out_proj, RMSNorm1(x))
        z = y + FF(RMSNorm2(x))

        return z
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_pdrop, residual_pdrop, weights):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights

    def forward(self, x):

        TokenEmbedding = torch.nn.Embedding(self.vocab_size, self.d_model)
        PositionEmbedding = torch.nn.Embedding(self.context_length, self.d_model)
        with torch.no_grad(): TokenEmbedding.weight.copy_(self.weights['token_embeddings.weight'])
        with torch.no_grad(): PositionEmbedding.weight.copy_(self.weights['position_embeddings.weight'])

        y = TokenEmbedding(x) + PositionEmbedding(torch.arange(x.shape[1]))


        for i in range(self.num_layers):
            Block = TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.attn_pdrop, self.residual_pdrop, 
                                     self.weights[f'layers.{i}.ln1.weight'], self.weights[f'layers.{i}.ln2.weight'], 
                                     self.weights[f'layers.{i}.ffn.w1.weight'], self.weights[f'layers.{i}.ffn.w2.weight'],
                                     self.weights[f'layers.{i}.attn.k_proj.weight'], self.weights[f'layers.{i}.attn.q_proj.weight'], 
                                     self.weights[f'layers.{i}.attn.v_proj.weight'], self.weights[f'layers.{i}.attn.output_proj.weight'])
            y = Block(y)

        RMS = RMSNorm(self.d_model, 1e-5, self.weights['ln_final.weight'])
        y = RMS(y)

        OutputEmbedding = torch.nn.Linear(self.d_model, self.vocab_size)
        with torch.no_grad(): OutputEmbedding.weight.copy_(self.weights['lm_head.weight'])
        y = OutputEmbedding(y)
        print(y.shape)
        y = softmax(y, dim=-1)

        return y