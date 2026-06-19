"""
Transformer model for 3D catheter tip position regression from MPI signals.
Supports both Standard (3 channels, 26929) and BA (2 channels, 80787) modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MultiHeadAttentionVariants(nn.Module):
    """Multi-head attention with standard and efficient variants."""
    def __init__(self, d_model, num_heads, variant='standard'):
        super().__init__()
        self.batch_first = True
        self.variant = variant
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        if variant == 'efficient':
            print("Using efficient attention variant")
            self.q_proj = nn.Linear(d_model, self.head_dim * num_heads)
            self.k_proj = nn.Linear(d_model, self.head_dim * num_heads)
            self.v_proj = nn.Linear(d_model, self.head_dim * num_heads)
        else:
            print("Using standard attention variant")
            self.qkv_proj = nn.Linear(d_model, d_model * 3) 

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        B, T, C = query.shape
        assert query.shape == key.shape == value.shape, "query/key/value shape mismatch"

        if self.variant == 'efficient':
            q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.q_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.q_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            qkv = self.qkv_proj(query).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores += attn_mask.unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        attn_output = (attn @ v).transpose(1, 2).reshape(B, T, C)
        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn
        return output

class CustomTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with variant attention."""
    def __init__(self, d_model, nhead, attn_variant='standard', dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = MultiHeadAttentionVariants(d_model, nhead, variant=attn_variant)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        if isinstance(src2, tuple):
            src2 = src2[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CatheterTransformer(nn.Module):
    """
    Transformer model for 3D catheter tip position regression from MPI signals.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dim_feedforward (int): Feedforward network dimension
        dropout (float): Dropout rate
        patch_size (int): Patch size for 1D convolution
        dropout_head (float): Dropout rate for head
        in_channels (int): Number of input channels (2 for BA, 3 for Standard)
        signal_len (int): Length of input signal
    """
    def __init__(self,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 patch_size: int = 32,
                 dropout_head: float = 0.0,
                 in_channels: int = 2,
                 signal_len: int = 80787):
        super().__init__()
        self.in_channels = in_channels
        self.signal_len = signal_len
        
        # 1D patch embedding conv
        self.patch_embed = nn.Conv1d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        
        # Calculate sequence length after patching
        self.seq_len = (signal_len - patch_size) // patch_size + 1
        self.flatten_dim = self.seq_len * embed_dim
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, dropout=dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim,
                                        nhead=num_heads,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Regression head
        self.fc1 = nn.Linear(self.flatten_dim, dim_feedforward)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_head)
        self.fc2 = nn.Linear(dim_feedforward, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Handle input shape
        if x.ndim == 2:
            # Flattened input (B, in_channels*signal_len)
            expected_flattened = batch_size * self.in_channels * self.signal_len
            if x.numel() != expected_flattened:
                raise ValueError(f"Expected {expected_flattened} elements for batch size {batch_size}, but got {x.numel()}")
            x = x.view(batch_size, self.in_channels, self.signal_len)
        elif x.ndim == 3:
            if x.shape[1] != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")
            if x.shape[2] != self.signal_len:
                raise ValueError(f"Expected signal length {self.signal_len}, got {x.shape[2]}")
        else:
            raise ValueError(f"Input must have shape (B, {self.in_channels}, {self.signal_len}) or (B, {self.in_channels}*{self.signal_len})")

        # Patch embedding
        x = self.patch_embed(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Regression head
        x = x.permute(1, 0, 2)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
