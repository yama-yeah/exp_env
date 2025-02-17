# from torch import nn
# import torch

# from exp_env.reservoir.base import BaseReservoirModule


# class ESNModule(nn.Module,BaseReservoirModule):
#     def __init__(self, in_features:int, out_features:int, max_log_len=None, alpha=0.1, connection_prob=0.05, activation_func=nn.Tanh(), initial_state:None|torch.Tensor=None,spectral_radius=0.99):
#         """
#         Initializes an Echo State Network (ESN) module.

#         Args:
#             in_features (int): Number of input features.
#             out_features (int): Number of output features.
#             max_log_len (int, optional): Maximum length of the reservoir state log. Defaults to None.
#             alpha (float, optional): Leakage rate for the reservoir state update. Defaults to 0.1.
#             connection_prob (float, optional): Probability of connection between reservoir neurons. Defaults to 0.05.
#             activation_func (torch.nn.Module, optional): Activation function for the reservoir neurons. Defaults to nn.Tanh().
#             initial_state (None or torch.Tensor, optional): Initial state of the reservoir. Defaults to None.
#         """
#         super().__init__()
#         self.spec_rad = spectral_radius
#         self.initial_state = initial_state
#         self.in_features = in_features
#         self.out_features = out_features
#         self.max_log_len = max_log_len
#         self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
#         self.connection_prob = connection_prob
#         self.activation_func = activation_func
#         self.w_in = torch.randn(self.out_features * self.in_features).reshape(self.in_features, self.out_features)
#         self.w_in = nn.Parameter(self.w_in, requires_grad=False)
#         while True: 
#             self.w_res = self.__create_weight(self.out_features)
#             #nanがあったらやり直す
#             if torch.isnan(self.w_res).sum()==0:
#                 break
#         self.w_res = nn.Parameter(self.w_res, requires_grad=False)

#         self.reset()

    
#     def __create_weight(self,hidden_dim:int):
#         w = torch.randn(hidden_dim**2).reshape(hidden_dim,hidden_dim)
#         w*=(torch.rand(hidden_dim,hidden_dim)<self.connection_prob)
#         spectral_radius = torch.abs(torch.linalg.eigvals(w)).max()
#         w=w/spectral_radius*self.spec_rad
#         return w
    
#     def next(self, x:torch.Tensor,current_state:torch.Tensor)->torch.Tensor:
#         u_in=x@self.w_in
#         l=(1-self.alpha)*current_state
#         r=self.alpha*self.activation_func(current_state@self.w_res+u_in)
#         return l+r
    
#     def add_log_state(self, state:torch.Tensor)->None:
#         if self.max_log_len is not None:
#             self.log_state.append(state)
#             if len(self.log_state)>self.max_log_len:
#                 self.log_state.pop(0)
#         else:
#             self.log_state.append(state)
    
#     def reset(self)->None:
#         if self.initial_state is not None:
#             self.log_state = [self.initial_state]
#         else:
#             self.log_state = [torch.zeros(self.out_features).to(self.w_res.device).to(self.w_res.dtype)]
    
#     def forward(self,X:torch.Tensor,one_step_mode=False,auto_reset=True)->torch.Tensor:
#         if one_step_mode:
#             self.add_log_state(self.next(X,self.log_state[-1]))
#             return self.log_state[-1]
#         else:
#             if auto_reset:
#                 self.reset()
#             X=X.permute(1, 0, 2)
#             for x in X:
#                 self.add_log_state(self.next(x,self.log_state[-1]))
#             return torch.stack(self.log_state[1:]).permute(1, 0, 2)

from torch import nn
import torch

from exp_env.reservoir.base import BaseReservoirModule
from exp_env.reservoir.esn import ESNModule


class DelayESNModule(ESNModule):
    def __init__(self, in_features:int, out_features:int, max_log_len=None, alpha=0.1, connection_prob=0.05, activation_func=nn.Tanh(), initial_state:None|torch.Tensor=None,spectral_radius=0.99,delay_num=1):
        """
        Initializes an Echo State Network (ESN) module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            max_log_len (int, optional): Maximum length of the reservoir state log. Defaults to None.
            alpha (float, optional): Leakage rate for the reservoir state update. Defaults to 0.1.
            connection_prob (float, optional): Probability of connection between reservoir neurons. Defaults to 0.05.
            activation_func (torch.nn.Module, optional): Activation function for the reservoir neurons. Defaults to nn.Tanh().
            initial_state (None or torch.Tensor, optional): Initial state of the reservoir. Defaults to None.
        """
        super().__init__(in_features, out_features, max_log_len, alpha, connection_prob, activation_func, initial_state, spectral_radius)
        self.delay_num=delay_num
        self.flag=True#True
    
    def forward(self,X:torch.Tensor,mask:list[list[int]],one_step_mode=False,auto_reset=True)->torch.Tensor:
        if one_step_mode:
            self.add_log_state(self.next(X,self.log_state[-1]))
            return self.log_state[-1]
        else:
            if auto_reset:
                self.reset()
            # T,B,X -> B,T,X
            #X=X.permute(1, 0, 2)
            X=X.transpose(0,1)
            x_size=X.size(1)
            # B,self.delay_num,self.out_features
            #B,D,M
            delay_state= torch.zeros(X.size(1),self.delay_num,self.out_features).to(X.device)
            self.input_log=torch.zeros(X.size(1),self.delay_num,X.size(-1)).to(X.device)
            if len(X.size())==4:
                #B,D,ICA,M
                delay_state=torch.zeros(X.size(1),self.delay_num,X.size(-2),self.out_features).to(X.device)
                self.input_log=torch.zeros(X.size(1),self.delay_num,X.size(-2),X.size(-1)).to(X.device)
            
            max_time=X.size(0)
            if len(mask)!=x_size:
                raise ValueError("mask length must be equal to X length")
            out_timings=[[] for _ in range(max_time)]
            for i in range(x_size):
                start=mask[i][0]
                end=mask[i][1]
                while_len=end-start
                for n in range(self.delay_num):
                    # n=0が0でパディングされる（コメント部分の説名）
                    # if n==0:
                    #     continue
                    out_timings[end-round(while_len/self.delay_num*n)-1].append((i,n))
            sum_X=torch.cumsum(X.detach().clone(),0)
            for t in range(max_time):
                x=X[t]
                self.add_log_state(self.next(x,self.log_state[-1]))
                flag=self.flag
                for i,n in out_timings[t]:
                    delay_state[i,n]=self.log_state[-1][i]
                    self.input_log[i,n]=sum_X[t][i]
                    if n!=self.delay_num-1:
                        self.input_log[i,n]-=self.input_log[i,n+1]
                    if flag:
                        self.log_state[-1][i]=torch.zeros(self.log_state[-1][i].size()).to(self.log_state[-1][i].device)
            # B,self.delay_num*self.out_features
            return delay_state.view(delay_state.size(0),-1)
