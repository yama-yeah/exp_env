import numpy as np
import torch
from torch import nn

# 3.2.1 Scaled Dot-Product Attention の実装
class ScaledDotProductAttention(nn.Module):
    def __init__(
            self,
            dimension_of_key: int, # =D_k
        ) -> None:
        super().__init__()
        self.scaler = np.sqrt(dimension_of_key)

    def forward(
        self,
        query: torch.Tensor,  # =Q
        key: torch.Tensor,  # =K
        value: torch.Tensor,  # =V
        mask: torch.Tensor|None = None,
    ) -> torch.Tensor:
        attention_weight = torch.matmul(query, torch.transpose(key, -1, -2)) # queryとkeyの類似度(内積)を計算
        attention_weight = attention_weight / self.scaler 
        # 高次元のベクトルはベクトル長が大きくなりやすい性質があるため、内積の値も大きくなりやすい
        # |a||b|cosθ = a・b より、a,bの長さが長いと内積は大きくなる
        # このため、内積をスケーリング（√d_kで割る）してあげることで、勾配消失を防ぐ
        
        

        if mask is not None: # maskに対する処理
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight=attention_weight.to('cpu')
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            ).to(key.device)
            # maskされた部分は類似度-infになり、softmaxで重み0として扱われる
            

        attention_weight = nn.functional.softmax(attention_weight, dim=2) # 類似度を合計1にするためにsoftmaxを取る
        return torch.matmul(attention_weight, value) # 類似度に基づいてvalueを重み付けして足し合わせる

# 3.2.2 Multi-Head Attention の実装
class MultiHeadAttention(nn.Module):
    def __init__(self, how_many_heads: int, dimension_of_query: int,dimension_of_key: int, dimension_of_value: int,dimension_of_head: int, dimension_of_output: int) -> None:
        super().__init__()
        self.h = how_many_heads
        self.d_q = dimension_of_query
        self.d_k = dimension_of_key
        self.d_v = dimension_of_value
        self.d_h = dimension_of_head
        self.d_o = dimension_of_output
        self.attention = ScaledDotProductAttention(dimension_of_key=self.d_k)
        self.W_q = nn.Linear(self.d_q, self.h * self.d_h, bias=False)
        self.W_k = nn.Linear(self.d_k, self.h * self.d_h, bias=False)
        self.W_v = nn.Linear(self.d_v, self.h * self.d_h, bias=False)
        self.W_o = nn.Linear(self.h * self.d_h, self.d_o, bias=False)
    
    def forward(
        self,
        query: torch.Tensor,  # =Q
        key: torch.Tensor,  # =K
        value: torch.Tensor,  # =V
        mask: torch.Tensor|None = None,
        cached_key: torch.Tensor|None = None,
        cached_value: torch.Tensor|None = None,
        position_encoder=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        # batch_size, seq_len, d_q-> batch_size, seq_len, h, d_hに写像
        key=self.W_k(key).view(batch_size, -1, self.h, self.d_h)
        query=self.W_q(query).view(batch_size, -1, self.h, self.d_h)
        value=self.W_v(value).view(batch_size, -1, self.h, self.d_h)

        start_pos = 0 if cached_key is None else cached_key.size(1)
        if position_encoder is not None:
            key = position_encoder(key, start_pos)
            query = position_encoder(query, start_pos)

        if cached_key is not None and cached_value is not None:
            key = torch.cat([cached_key, key], dim=1)
            value = torch.cat([cached_value, value], dim=1)

        
        
        # maskとkey,query,valueの次元を合わせる
        key= key.transpose(1, 2) # batch_size, h, all_seq_len, d_h
        query = query.transpose(1, 2) # batch_size, h, seq_len, d_h
        value = value.transpose(1, 2) # batch_size, h, all_seq_len, d_h

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1) # batch_size, h, seq_len, seq_len
        
        hidden_state:torch.Tensor = self.attention(query, key, value, mask) # batch_size, h, seq_len, d_h

        # maskに合わせた次元を元に戻す
        hidden_state = hidden_state.transpose(1, 2) # batch_size, seq_len, h, d_h
        hidden_state = hidden_state.contiguous().view(batch_size, -1, self.h * self.d_h) # batch_size, seq_len, h*d_h
        # hidden_state,cached_key,cached_valueを返す
        hidden_state=self.W_o(hidden_state) # batch_size, seq_len, d_o
        return hidden_state, key, value






