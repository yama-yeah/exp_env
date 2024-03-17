from torch import nn
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
        return F.l1_loss(y_hat, y)
    
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
        return F.l1_loss(y_hat, y)
    
    def forward(self, x):
        return self.linear(x)