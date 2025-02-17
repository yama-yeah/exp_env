import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
        query: torch.Tensor,  # =Q (batch_size, h, seq_len, d_h)
        key: torch.Tensor,  # =K (batch_size, h, all_seq_len, d_h)
        value: torch.Tensor,  # =V (batch_size, h, all_seq_len, d_h)
        mask: torch.Tensor|None = None,
    ) -> torch.Tensor:
        # (batch_size, h, seq_len, d_h) x (batch_size, h, d_h, all_seq_len) = (batch_size, h, seq_len, all_seq_len)
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
            attention_weight=attention_weight#.to('cpu')
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(attention_weight.dtype).max
            )
            #.to(key.device)
            # maskされた部分は類似度-infになり、softmaxで重み0として扱われる
            

        attention_weight = nn.functional.softmax(attention_weight, dim=-1) # 類似度を合計1にするためにsoftmaxを取る
        return torch.matmul(attention_weight, value) # 類似度に基づいてvalueを重み付けして足し合わせる batch_size, h, seq_len, d_h
    
class  LinearAttention (nn.Module): 
    def  __init__ ( self ,activation = F.elu,eps = 1e-6,is_causal=False):
        super (LinearAttention, self).__init__() 
        self.eps = eps
        self.activation = activation
        self.is_causal=is_causal

    def  elu_feature_map ( self, x ): 
        return self.activation(x) +  1

    def  forward ( self, Q, K, V, mask = None):
        # Q<= batch, heads, query_len, d_model
        # K<= batch, heads, all_seq_len, d_model
        # V<= batch, heads, all_seq_len, d_model
        # mask<=batch, heads, query_len, all_seq_len
        Q = self.elu_feature_map(Q) 
        K = self.elu_feature_map(K) 
        # 3.3 casual masking
        start_pos=K.size(2)-Q.size(2)
        end_pos=K.size(2)
        is_causal=False
        with torch.no_grad():
            if mask is not None:
                for i in range(mask.size(0)):
                    if mask[i,0,0].float().sum(dim=-1)==mask.size(-1):
                        i+=1
                    else:
                        is_causal=mask[i,0,-1].float().sum(dim=-1)!=mask[i,0,0].float().sum(dim=-1)
                        break
                mask=mask[:,0,-1,:].squeeze().unsqueeze(1)
                mask=mask.repeat(1,K.shape[1],1).unsqueeze(-1)
                # K=K.reshape(K.size(0),K.size(1)*K.size(2),-1)
                K=K*mask
                # K=K.reshape(V.size(0),V.size(1),V.size(2),-1)
                # V=V.reshape(V.size(0),V.size(1)*V.size(2),-1)
                V=V*mask
                del mask
                # V=V.reshape(V.size(0),V.size(1),V.size(2),-1)
        if is_causal:
            V=self.causal_forward(Q, K, V,start_pos,end_pos)
        else:
            V=self._forward(Q, K, V)
        return V.contiguous()
    def causal_forward(self,Q, K, V,start_pos,end_pos):
        #KV <= batch, heads, query_len　特定の位置に至るまでのkeyとvalueの内積の和
        KV=torch.einsum( "bhsd,bhsd->bhs" , K, V) #batch, heads, all_seq_len
        KV=torch.cumsum(KV, dim=-1)[:,:,start_pos:end_pos] #batch, heads, query_len
        

        #Z<= batch, heads, query_len 特定の位置に至るまでのkeyの和とqueryの内積の逆数
        K=torch.cumsum(K, dim=-1)[:,:,start_pos:end_pos] #batch, heads, query_len
        Z=1/(torch.einsum( "bhld,bhld->bhl" , Q, K)+self.eps)

        V=torch.einsum( "bhld,bhs,bhl->bhld" , Q, KV, Z) #batch, heads, query_len, d_model
        return V.contiguous()
    def _forward(self,Q, K, V):
        #keyとvalueの内積の和
        #KV <= batch, heads
        KV=torch.einsum( "bhsd,bhsd->bh" , K, V)

        #keyの和とqueryの内積の逆数
        #Z<= batch, heads, query_len
        Z=1/(torch.einsum( "bhld,bhd->bhl" , Q, K.sum(dim=-2))+self.eps)
        #queryとkeyの関係からKVの持つ値を操作している
        V=torch.einsum( "bhld,bh,bhl->bhld" , Q, KV, Z)
        return V.contiguous()
SAT=LinearAttention
# SAT=ScaledDotProductAttention
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
        if SAT==LinearAttention:
            self.attention = LinearAttention()
        else:
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
            cached_key=cached_key.to(key.device)
            cached_value=cached_value.to(key.device)
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






