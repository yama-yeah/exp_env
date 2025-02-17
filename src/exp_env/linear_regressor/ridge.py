from torch import nn
import torch.nn.functional as F
import torch

from exp_env.linear_regressor.base import BaseLinearModel

class LazyRidge(nn.Module,BaseLinearModel):
    def __init__(self, out_features,alpha=1.0,useGradientDescent=False):
        super().__init__()
        self.alpha = alpha
        self.linear = nn.LazyLinear(out_features=out_features)
        self.linear.weight.requires_grad = useGradientDescent
        self.useGradientDescent = useGradientDescent
    
    def fit(self, X, y):
        if self.useGradientDescent:
            raise Exception('if you wanna use gradient descent, please use torch.optim')
        with torch.no_grad():
            I=torch.eye(X.shape[1])
            lambdaI=self.alpha*I
            WB=torch.inverse(X.T@X+lambdaI)@X.T@y
            self.linear.weight.data=WB[:-1]
            self.linear.bias.data=WB[-1]
            del WB,I,lambdaI
            torch.cuda.empty_cache()
    
    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y) + self.alpha * F.mse_loss(self.linear.weight, torch.zeros_like(self.linear.weight))

    def forward(self, x):
        return self.linear(x)

class Ridge(nn.Module,BaseLinearModel):
    def __init__(self, in_features,out_features,alpha=1.0,useGradientDescent=False):
        super().__init__()
        self.alpha = alpha
        self.linear = nn.Linear(in_features,out_features=out_features)
        self.linear.weight.requires_grad = useGradientDescent
        self.useGradientDescent = useGradientDescent

    def fit(self, X:torch.Tensor, y,use_bias=True):
        if self.useGradientDescent:
            raise Exception('if you wanna use gradient descent, please use torch.optim')
        with torch.no_grad():
            if use_bias:
                one=torch.ones((X.shape[0],1)).to(X.device).to(X.dtype)
                X=torch.cat([X,one],dim=-1)
            I=torch.eye(X.shape[1]).to(X.device).to(X.dtype)
            lambdaI=self.alpha*I
            a=X.T@X+lambdaI
            WB=torch.inverse(a)@X.T
            WB=WB@y
            if use_bias:
                self.linear.weight.data=WB[:-1].T
                self.linear.bias.data=WB[-1]
            else:
                self.linear.weight.data=WB.T
                self.linear.bias.data=torch.zeros((WB.shape[1])).to(X.device).to(X.dtype)
            del WB,I,lambdaI
            torch.cuda.empty_cache()

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y) + self.alpha * F.mse_loss(self.linear.weight, torch.zeros_like(self.linear.weight))
    
    def forward(self, x):
        return self.linear(x)