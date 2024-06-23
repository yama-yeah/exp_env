import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len,theta=10000.0):
        super(RotaryPositionalEmbedding, self).__init__()
        self.freq_cis=self.precompute_freqs_cis(d_model, max_seq_len, theta)

    def precompute_freqs_cis(self,dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def reshape_for_broadcast(self,freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, x_).to(x_.device)
        xq_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return xq_out.type_as(x)
    
    def forward(self, x: torch.Tensor,start_pos:int=0):
        seq_len = x.size(1)
        x = self.apply_rotary_emb(x, self.freq_cis[start_pos:start_pos+seq_len])
        return x