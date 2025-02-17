import torch
def quantize_relu(x,Z, n):
    max_val = 2.**n - 1.
    return x.clamp(Z, max_val).round()


def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)
    scale = max(scale, 1e-6)  # avoid dividing by 0

    z= (x - min_val) / scale + qmin
    z = z.clamp(qmin, qmax).round()

    n= torch.round(torch.log2(scale))

    q=torch.round(x / scale)+z
    q=q.clamp(qmin, qmax)
    return q, z, n

def dequantize_tensor(q, z, n):
    return 2.**n * (q-z)

def multiply_quantize_tensors(x, y, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)
    scale = max(scale, 1e-6)  # avoid dividing by 0

    z= (x - min_val) / scale + qmin
    z = z.clamp(qmin, qmax).round()

    n= torch.round(torch.log2(scale))

    q=torch.round(x / scale)+z
    q=q.clamp(qmin, qmax)

    z2= (y - min_val) / scale + qmin
    z2 = z2.clamp(qmin, qmax).round()

    q2=torch.round(y / scale)+z2
    q2=q2.clamp(qmin, qmax)

    return q*q2, z, n

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    q, z, n = quantize_tensor(x)
    print(q, z, n)