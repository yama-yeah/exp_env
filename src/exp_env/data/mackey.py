import numpy as np
from exp_env.data.data_maker_base import DataMaker, DataMakerSpec
class MackeySpec(DataMakerSpec):
    def __init__(self, max_T, dt, how_many, cut_off=0,tau=17):
        super().__init__()
        self.max_T = max_T
        self.dt = dt
        self.how_many = how_many
        self.cut_off = cut_off

class MackeyMaker(DataMaker):
    def __init__(self, spec: MackeySpec):
        self.spec = spec
    
    def mackey(self,X,tau,beta=0.2,gamma=0.1,n=10):
        tau/=self.spec.dt
        tau=int(tau)
        if X.shape[1]<=tau:
            new_X= -X[:,-1]*gamma
            return new_X*self.spec.dt+X[:,-1]
        else:
            new_X=-X[:,-1]*gamma+beta*X[:,-tau-1]/(1+X[:,-tau-1]**n)
            return X[:,-1]+new_X*self.spec.dt

    def make(self,tau=17,beta=0.2,gamma=0.1,n=10):
        X=np.random.rand(self.spec.how_many,1)*4-2
        for i in range(self.spec.max_T):
            new_X=self.mackey(X,tau,beta,gamma,n).reshape(-1,1)
            X=np.concatenate([X,new_X],axis=1)
        return X[:,self.spec.cut_off:]

if __name__ == '__main__':
    spec = MackeySpec(1000,0.5,1000)
    maker = MackeyMaker(spec)
    data = maker.make()
    print(data.shape)