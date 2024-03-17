from torch import nn
import torch

from exp_env.reservoir.base import BaseReservoirModule


class ESNModule(nn.Module,BaseReservoirModule):
    def __init__(self, in_features:int, out_features:int,max_log_len=None, alpha=0.1,connection_prob=0.05,activation_func=nn.Tanh(),initial_state:None|torch.Tensor=None):
        super().__init__()
        self.initial_state = initial_state
        self.in_features = in_features
        self.out_features = out_features
        self.reset()
        self.max_log_len = max_log_len
        self.alpha = alpha
        self.connection_prob = connection_prob
        self.activation_func = activation_func
        self.w_in= torch.randn(self.out_features*self.in_features).reshape(self.out_features,self.in_features)
        self.w_in= nn.Parameter(self.w_in,requires_grad=False)
        self.w_res = self.__create_weight(self.out_features)
        self.w_res= nn.Parameter(self.w_res,requires_grad=False)
    
    def __create_weight(self,hidden_dim:int):
        w = torch.randn(hidden_dim**2).reshape(hidden_dim,hidden_dim)
        w*=(torch.rand(hidden_dim,hidden_dim)<self.connection_prob)
        spectral_radius = torch.abs(torch.linalg.eigvals(w)).max()
        w=w/spectral_radius*0.99
        return w
    
    def next(self, x:torch.Tensor,current_state:torch.Tensor)->torch.Tensor:
        u_in=self.w_in@x
        return (1-self.alpha)*current_state+self.alpha*self.activation_func(self.w_res@current_state+u_in)
    
    def add_log_state(self, state:torch.Tensor)->None:
        if self.max_log_len is not None:
            self.log_state.append(state)
            if len(self.log_state)>self.max_log_len:
                self.log_state.pop(0)
        else:
            self.log_state.append(state)
    
    def reset(self)->None:
        if self.initial_state is not None:
            self.log_state = [self.initial_state]
        else:
            self.log_state = [torch.zeros(self.out_features)]
    
    def forward(self,X:torch.Tensor,one_step_mode=False)->torch.Tensor:
        if one_step_mode:
            self.add_log_state(self.next(X,self.log_state[-1]))
            return self.log_state[-1]
        else:
            self.reset()
            X=X.permute(1, 0, 2)
            for x in X:
                self.add_log_state(self.next(x,self.log_state[-1]))
            return torch.stack(self.log_state).permute(1, 0, 2)

