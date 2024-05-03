from exp_env.linear_regressor.base import BaseLinearModel
from torch import Tensor
from torch.nn import functional as F

class PreTrainedLinear(BaseLinearModel):
    def __init__(self, weight: Tensor, bias: Tensor|None = None):
        super().__init__()
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        if bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def fit(self, X: Tensor, y: Tensor):
        raise Exception('PreTrainedLinear is a pre-trained model, fit method is not supported')

    def loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        raise Exception('PreTrainedLinear is a pre-trained model, loss method is not supported')