from torch import nn
import torch
import torch.nn.functional as F

from exp_env.linear_regressor.base import BaseLinearModel

class LazyLasso(nn.Module,BaseLinearModel):
    def __init__(self, out_features,alpha=1.0,useGradientDescent=False):
        super().__init__()
        self.alpha = alpha
        self.linear = nn.LazyLinear(out_features=out_features)
        self.linear.weight.requires_grad = useGradientDescent
        self.useGradientDescent = useGradientDescent

    def fit(self, X, y):
        if self.useGradientDescent:
            raise Exception('if you wanna use gradient descent, please use torch.optim')
        else:
            raise Exception('l1 loss can not be solved by inverse matrix')

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y) + self.alpha * F.l1_loss(self.linear.weight, torch.zeros_like(self.linear.weight))
    
    def forward(self, x):
        return self.linear(x)
        
class Lasso(nn.Module,BaseLinearModel):
    def __init__(self, in_features,out_features,alpha=1.0,useGradientDescent=True):
        super().__init__()
        self.alpha = alpha
        self.linear = nn.Linear(in_features,out_features=out_features)
        self.linear.weight.requires_grad = useGradientDescent
        self.useGradientDescent = useGradientDescent

    def fit(self, X, y):
        if self.useGradientDescent:
            raise Exception('if you wanna use gradient descent, please use torch.optim')
        else:
            raise Exception('l1 loss can not be solved by inverse matrix')

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y) + self.alpha * F.l1_loss(self.linear.weight, torch.zeros_like(self.linear.weight))
    
    def elastic_loss(self, y_hat, y,alpha2=0.5):
        return F.mse_loss(y_hat, y) + self.alpha * F.l1_loss(self.linear.weight, torch.zeros_like(self.linear.weight)) + alpha2 * F.mse_loss(self.linear.weight, torch.zeros_like(self.linear.weight))
    
    def forward(self, x):
        return self.linear(x)