import torch


def weight_quant(w: torch.Tensor):
    #入力を-1から1まで平均０の正規分布と仮定
    scale=1#スケール1
    e = 0#平均0
    u = (w - e).sign()*scale
    u = u.clamp(0, 1).round().to(torch.bool)
    #整数化されたuを返す
    return u
def weight_quant_one_point_five(w: torch.Tensor):
    #入力を-1から1まで平均０の正規分布と仮定
    max_w = w.max()
    min_w = w.min()
    w=(w-min_w)/(max_w-min_w)*2-1
    w=w.clamp(-1,1).round()
    return w.to(torch.int8)
def activation_quant(x: torch.Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    #入力はLayerNormにより最大値が1、最小値が-1、平均が0になっていると仮定
    scale = 127.
    y = (x * scale).round().clamp_(-127, 127)
    return y.to(torch.bfloat16)
def x_quantize(x: torch.Tensor):
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    x_quant = activation_quant(x_norm)
    return x_quant

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    q= weight_quant(x).to(torch.bfloat16)
    print(q)