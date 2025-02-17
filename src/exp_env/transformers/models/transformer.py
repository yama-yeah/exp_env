import torch
import torch.nn as nn
from exp_env.transformers.modules.attention import MultiHeadAttention
from exp_env.transformers.modules.dropout import Dropouts
from exp_env.transformers.modules.ffn import FeedForward
from exp_env.transformers.modules.position_emb import RotaryPositionalEmbedding

MDS=2
PRE=False

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

class EncoderBlock(nn.Module):
    def __init__(self,how_many_heads: int,dimension_of_query: int,hidden_dim: int,layer_norm_eps: float,dp_rate: float = 0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.attention = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, hidden_dim// how_many_heads, dimension_of_query)
        self.ffn = FeedForward(dimension_of_query, hidden_dim)
        self.dropout = Dropouts(dp_rate,MDS)
        self.att_drop = Dropouts(dp_rate,MDS)
        self.pe = None # RotaryPositionalEmbedding(hidden_dim// how_many_heads, 256)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        y, _, _ = self.attention(x, x, x, mask,position_encoder=self.pe)
        y = self.att_drop(y)
        z = self.layer_norm1(y+x)
        
        y = self.ffn(z)
        y = self.dropout(y)
        x = self.layer_norm2(x+y+z)
        return x
    
# class EncoderBlock(nn.Module):
#     def __init__(self,how_many_heads: int,dimension_of_query: int,hidden_dim: int,layer_norm_eps: float,dp_rate: float = 0.1):
#         super().__init__()
#         self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
#         self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
#         self.attention = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, hidden_dim// how_many_heads, dimension_of_query)
#         self.ffn = FeedForward(dimension_of_query, hidden_dim)
#         self.dropout = Dropouts(dp_rate,MDS)
#         self.att_drop = Dropouts(dp_rate,MDS)
#         self.pe = None # RotaryPositionalEmbedding(hidden_dim// how_many_heads, 256)
    
#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         y = self.layer_norm1(x)
#         y, _, _ = self.attention(y, y, y, mask,position_encoder=self.pe)
#         y = self.att_drop(y)
#         z = self.layer_norm2(y+x)
        
#         y = self.ffn(z)
#         y = self.dropout(y)
#         return x + y + z

class Encoder(nn.Module):
    def __init__(
        self,
        how_many_block: int,
        how_many_heads: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_idx: int,
        layer_norm_eps=1e-5,
        max_seq_len=512,
        position_embedding = RotaryPositionalEmbedding,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = position_embedding(embedding_dim, max_seq_len)
        self.block_list = torch.nn.ModuleList()
        max_drop=0.1
        for i in range(how_many_block):
            self.block_list.append(
                EncoderBlock(how_many_heads, embedding_dim, hidden_dim, layer_norm_eps,max_drop*(i+1)/how_many_block),
            )
        if PRE:
            self.norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_embedding(x)
        for block in self.block_list:
            x = block(x, mask)
        if PRE:
            x = self.norm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,how_many_heads: int,dimension_of_query: int,hidden_dim:int,layer_norm_eps: float,use_cache: bool = False,dp_rate: float = 0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.attention1 = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, hidden_dim// how_many_heads, dimension_of_query)
        self.attention2 = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, hidden_dim// how_many_heads, dimension_of_query)
        self.ffn = FeedForward(dimension_of_query, hidden_dim)
        self.dropout = Dropouts(dp_rate,MDS)
        self.att_drop1 = Dropouts(dp_rate,MDS)
        self.att_drop2 = Dropouts(dp_rate,MDS)
        self.cache_key1=None
        self.cache_value1=None
        self.cache_key2=None
        self.cache_value2=None
        self.use_cache = use_cache
        self.pe = None # RotaryPositionalEmbedding(hidden_dim// how_many_heads, 256)

    def reset_cache(self):
        del self.cache_key1,self.cache_value1,self.cache_key2,self.cache_value2
        self.cache_key1=None
        self.cache_value1=None
        self.cache_key2=None
        self.cache_value2=None

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _,_ = self.attention1(x, x, x, mask,cached_key=self.cache_key1,cached_value=self.cache_value1,position_encoder=self.pe)
        y = self.att_drop1(y)
        # self.cache_key1 = self.cache_key1.to('cpu')
        # self.cache_value1 = self.cache_value1.to('cpu')
        z = self.layer_norm1(y+x)
        y, _,_= self.attention2(z, encoder_output, encoder_output, mask,cached_key=self.cache_key2,cached_value=self.cache_value2,position_encoder=self.pe)
        y = self.att_drop2(y)
        # self.cache_key2 = self.cache_key2.to('cpu')
        # self.cache_value2 = self.cache_value2.to('cpu')
        z = self.layer_norm2(y+z)
        y = self.ffn(z)
        y = self.dropout(y)
        x = self.layer_norm3(x+y+z)
        # if not self.use_cache:
        #     self.reset_cache()
        return x

# class DecoderBlock(nn.Module):
#     def __init__(self,how_many_heads: int,dimension_of_query: int,hidden_dim:int,layer_norm_eps: float,use_cache: bool = False,dp_rate: float = 0.1):
#         super().__init__()
#         self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
#         self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
#         self.layer_norm3 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
#         self.attention1 = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, hidden_dim// how_many_heads, dimension_of_query)
#         self.attention2 = MultiHeadAttention(how_many_heads, dimension_of_query, dimension_of_query, dimension_of_query, hidden_dim// how_many_heads, dimension_of_query)
#         self.ffn = FeedForward(dimension_of_query, hidden_dim)
#         self.dropout = Dropouts(dp_rate,MDS)
#         self.att_drop1 = Dropouts(dp_rate,MDS)
#         self.att_drop2 = Dropouts(dp_rate,MDS)
#         self.cache_key1=None
#         self.cache_value1=None
#         self.cache_key2=None
#         self.cache_value2=None
#         self.use_cache = use_cache
#         self.pe = None # RotaryPositionalEmbedding(hidden_dim// how_many_heads, 256)

#     def reset_cache(self):
#         del self.cache_key1,self.cache_value1,self.cache_key2,self.cache_value2
#         self.cache_key1=None
#         self.cache_value1=None
#         self.cache_key2=None
#         self.cache_value2=None

#     def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         y = self.layer_norm1(x)
#         y, self.cache_key1, self.cache_value1 = self.attention1(y, y, y, mask,cached_key=self.cache_key1,cached_value=self.cache_value1,position_encoder=self.pe)
#         y = self.att_drop1(y)
#         # self.cache_key1 = self.cache_key1.to('cpu')
#         # self.cache_value1 = self.cache_value1.to('cpu')
#         z = self.layer_norm2(y+x)
#         y, self.cache_key2, self.cache_value2 = self.attention2(z, encoder_output, encoder_output, mask,cached_key=self.cache_key2,cached_value=self.cache_value2,position_encoder=self.pe)
#         y = self.att_drop2(y)
#         # self.cache_key2 = self.cache_key2.to('cpu')
#         # self.cache_value2 = self.cache_value2.to('cpu')
#         z = self.layer_norm3(y+z)
#         y = self.ffn(z)
#         y = self.dropout(y)
#         if not self.use_cache:
#             self.reset_cache()
#         return x + y + z

class Decoder(nn.Module):
    def __init__(
        self,
        how_many_block: int,
        how_many_heads: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
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
        max_drop=0.1
        for i in range(how_many_block):
            self.block_list.append(
                DecoderBlock(how_many_heads, embedding_dim,hidden_dim, layer_norm_eps, use_cache, max_drop),
            )
        if PRE:
            self.norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_embedding(x)
        for block in self.block_list:
            x = block(x, encoder_output, mask)
        if PRE:
            x = self.norm(x)
        x = self.linear(x)
        #x = self.softmax(x)
        return x

