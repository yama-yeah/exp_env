import torch
import torch.nn as nn
from exp_env.transformers.modules.attention import MultiHeadAttention
from exp_env.transformers.modules.ffn import FeedForward
from exp_env.transformers.modules.position_emb import RotaryPositionalEmbedding

class EncoderBlock(nn.Module):
    def __init__(self,how_many_heads: int,dimension_of_query: int,layer_norm_eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.attention = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, dimension_of_query// how_many_heads, dimension_of_query)
        self.ffn = FeedForward(dimension_of_query, dimension_of_query)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        y, _, _ = self.attention(x, x, x, mask)
        x = self.layer_norm1(y+x)
        
        y = self.ffn(x)
        x = self.layer_norm2(x+y)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        how_many_block: int,
        how_many_heads: int,
        vocab_size: int,
        embedding_dim: int,
        pad_idx: int,
        layer_norm_eps=1e-5,
        max_seq_len=512,
        position_embedding = RotaryPositionalEmbedding,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = position_embedding(embedding_dim, max_seq_len)
        self.block_list = torch.nn.ModuleList()
        for _ in range(how_many_block):
            self.block_list.append(
                EncoderBlock(how_many_heads, embedding_dim, layer_norm_eps)
            )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_embedding(x)
        for block in self.block_list:
            x = block(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,how_many_heads: int,dimension_of_query: int,layer_norm_eps: float,use_cache: bool = False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.attention1 = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, dimension_of_query// how_many_heads, dimension_of_query)
        self.attention2 = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, dimension_of_query// how_many_heads, dimension_of_query)
        self.ffn = FeedForward(dimension_of_query, dimension_of_query)
        self.cache_key1=None
        self.cache_value1=None
        self.cache_key2=None
        self.cache_value2=None
        self.use_cache = use_cache

    def reset_cache(self):
        self.cache_key1=None
        self.cache_value1=None
        self.cache_key2=None
        self.cache_value2=None

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, self.cache_key1, self.cache_value1 = self.attention1(x, x, x, mask,cached_key=self.cache_key1,cached_value=self.cache_value1)
        x = self.layer_norm1(y+x)
        y, self.cache_key2, self.cache_value2 = self.attention2(x, encoder_output, encoder_output, mask,cached_key=self.cache_key2,cached_value=self.cache_value2)
        x = self.layer_norm2(y+x)
        y = self.ffn(x)
        x = self.layer_norm3(x+y)
        if not self.use_cache:
            self.reset_cache()
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        how_many_block: int,
        how_many_heads: int,
        vocab_size: int,
        embedding_dim: int,
        pad_idx: int,
        layer_norm_eps=1e-5,
        max_seq_len=512,
        position_embedding = RotaryPositionalEmbedding,
        use_cache: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = position_embedding(embedding_dim, max_seq_len)
        self.block_list = torch.nn.ModuleList()
        for _ in range(how_many_block):
            self.block_list.append(
                DecoderBlock(how_many_heads, embedding_dim, layer_norm_eps, use_cache)
            )
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_embedding(x)
        for block in self.block_list:
            x = block(x, encoder_output, mask)
        x = self.linear(x)
        x = self.softmax(x)
        return x

