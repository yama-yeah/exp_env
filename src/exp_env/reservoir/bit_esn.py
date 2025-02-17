from torch import nn
import torch
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
from exp_env.reservoir.base import BaseReservoirModule
from exp_env.quantization.bitnet import weight_quant_one_point_five,activation_quant,x_quantize

class BitESNModule(nn.Module,BaseReservoirModule):

    def __init__(self, in_features:int, out_features:int, max_log_len=None, alpha=0.1, connection_prob=0.05, activation_func=nn.Tanh(), initial_state:None|torch.Tensor=None,spectral_radius=0.99):
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
        super().__init__()
        self.spec_rad = spectral_radius
        self.initial_state = initial_state
        self.in_features = in_features
        self.out_features = out_features
        self.max_log_len = max_log_len
        # alpha*=127
        # alpha=alpha.clamp(0,127)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.connection_prob = connection_prob
        self.activation_func = activation_func
        self.w_in = torch.randn(self.out_features * self.in_features).reshape(self.in_features, self.out_features)
        self.w_in = nn.Parameter(self.w_in, requires_grad=False)
        while True: 
            self.__w_res = self.__create_weight(self.out_features)
            
            #nanがあったらやり直す
            if torch.isnan(self.__w_res).sum()==0:
                self.__w_res = weight_quant_one_point_five(self.__w_res)
                break
            
        self.__w_res = nn.Parameter(self.__w_res, requires_grad=False)

        self.reset()

    @property
    def w_res(self):
        return self.__w_res.sign().to(torch.bfloat16)
    @w_res.setter
    def w_res(self, w):
        self.__w_res = w

    
    def __create_weight(self,hidden_dim:int):
        w = torch.randn(hidden_dim**2).reshape(hidden_dim,hidden_dim)
        w*=(torch.rand(hidden_dim,hidden_dim)<self.connection_prob)
        spectral_radius = torch.abs(torch.linalg.eigvals(w)).max()
        w=w/spectral_radius*self.spec_rad
        return w
    
    def next(self, x:torch.Tensor,current_state:torch.Tensor)->torch.Tensor:
        
        u_in=x@self.w_in
        # u_in=activation_quant(u_in)#scaleは線形回帰に推論させる
        l=(1-self.alpha)*current_state
        i=x_quantize(current_state)@self.w_res
        i=i.to(torch.float32)/127
        i=i+u_in
        r=self.alpha*self.activation_func(i)
        return l+r
    
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
            self.log_state = [torch.zeros(self.out_features).to(self.w_res.device).to(self.w_res.dtype)]
    
    def forward(self,X:torch.Tensor,one_step_mode=False)->torch.Tensor:
        if one_step_mode:
            self.add_log_state(self.next(X,self.log_state[-1]))
            return self.log_state[-1]
        else:
            X=X.permute(1, 0, 2)
            for x in X:
                self.add_log_state(self.next(x,self.log_state[-1]))
            logs=torch.stack(self.log_state[1:])
            logs=logs
            return logs.permute(1, 0, 2).to(torch.float32)
        

class BitESN158Module(nn.Module,BaseReservoirModule):

    def __init__(self, in_features:int, out_features:int, max_log_len=None, alpha=0.1, connection_prob=0.05, activation_func=nn.Tanh(), initial_state:None|torch.Tensor=None,spectral_radius=0.99):
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
        super().__init__()
        self.spec_rad = spectral_radius
        self.initial_state = initial_state
        self.in_features = in_features
        self.out_features = out_features
        self.max_log_len = max_log_len
        # alpha*=127
        # alpha=alpha.clamp(0,127)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.connection_prob = connection_prob
        self.activation_func = activation_func
        self.w_in = torch.randn(self.out_features * self.in_features).reshape(self.in_features, self.out_features)
        self.w_in = nn.Parameter(self.w_in, requires_grad=False)
        while True: 
            w_res = self.__create_weight(self.out_features)
            
            #nanがあったらやり直す
            if torch.isnan(self.w_res).sum()==0:
                self.w_res = weight_quant_one_point_five(w_res)
                break
            
        self.w_res = nn.Parameter(self.w_res, requires_grad=False)

        self.reset()

    @property
    def w_res(self):
        return self.__w_res.to(torch.bfloat16)
    @w_res.setter
    def w_res(self, w):
        self.__w_res = w

    
    def __create_weight(self,hidden_dim:int):
        w = torch.randn(hidden_dim**2).reshape(hidden_dim,hidden_dim)
        w*=(torch.rand(hidden_dim,hidden_dim)<self.connection_prob)
        spectral_radius = torch.abs(torch.linalg.eigvals(w)).max()
        w=w/spectral_radius*self.spec_rad
        return w
    
    def next(self, x:torch.Tensor,current_state:torch.Tensor)->torch.Tensor:
        
        u_in=x@self.w_in
        # u_in=activation_quant(u_in)#scaleは線形回帰に推論させる
        l=(1-self.alpha)*current_state
        i=x_quantize(current_state)@self.w_res
        i=i.to(torch.float32)/127
        i=i+u_in
        r=self.alpha*self.activation_func(i)
        return l+r
    
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
            self.log_state = [torch.zeros(self.out_features).to(self.w_res.device)]
    
    def forward(self,X:torch.Tensor,one_step_mode=False)->torch.Tensor:
        if one_step_mode:
            self.add_log_state(self.next(X,self.log_state[-1]))
            return self.log_state[-1]
        else:
            X=X.permute(1, 0, 2)
            for x in X:
                self.add_log_state(self.next(x,self.log_state[-1]))
            logs=torch.stack(self.log_state[1:])
            logs=logs
            return logs.permute(1, 0, 2).to(torch.float32)
