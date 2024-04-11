from typing import Callable, Iterator, Optional
import numpy as np
import torch
from exp_env.data.data_maker_base import CycleDataSliderSpec, CycleDataSlider
from dataclasses import dataclass
import random
@dataclass
class TriangleFunc:
    sin=np.sin
    cos=np.cos
    tan=np.tan

class TriangleSliderSpec(CycleDataSliderSpec):
    func: TriangleFunc
    freq_range: tuple
    def __init__(self, cycle_T: int, slide: int, x_func:Callable[[np.ndarray],np.ndarray], y_func:Callable[[np.ndarray],np.ndarray] ,freq_range: tuple, how_many: Optional[int] = 1, length: Optional[int] = None):
        super().__init__(cycle_T, slide, how_many, length)
        self.x_func = x_func
        self.y_func = y_func
  

class Triangle(CycleDataSlider):
    def __init__(self, spec: TriangleSliderSpec):
        self.spec = spec

    def make_random(self,how_many:int):
        datas=[]
        #ユニークなデータを作成
        while True:
            data=random.random()
            if data not in datas:
                datas.append(data)
            if len(datas)==how_many:
                break
        return datas

    def linspace(self,random_start:float):
        T=np.pi*2
        start=random_start+T
        d_theta=T/self.spec.cycle_T
        end=start+T*self.spec.length*d_theta
        length=self.spec.length
        slide=self.spec.slide
        num=length+slide
        return np.linspace(start,end,num)
    
    def generate_randoms(self,how_many:int):
        randoms=self.make_random(how_many)
        for random_start in randoms:
            yield random_start

    def generate_linespaces(self,randoms:Iterator[float]):
        for random_start in randoms:
            yield self.linspace(random_start)
    
    def split_x_y(self,linspaces:Iterator[np.ndarray]):
        for linspace in linspaces:
            x_range=linspace[:self.spec.length]
            y_range=linspace[self.spec.slide:]
            yield x_range,y_range
    def adapt_func(self,xy_range:Iterator[tuple]):
        for x_range,y_range in xy_range:
            x=self.spec.x_func(x_range)
            y=self.spec.y_func(y_range)
            yield x,y
    def add_noise(self,xy:Iterator[tuple]):
        for x,y in xy:
            x+=np.random.normal(size=x.shape[0])*0.05
            yield x,y
    
    
    
    def make(self,how_many:int)->tuple[np.ndarray,np.ndarray]:
        randoms=self.generate_randoms(how_many)
        linspaces=self.generate_linespaces(randoms)
        xy=self.split_x_y(linspaces)
        xy=self.adapt_func(xy)
        # xy=self.add_noise(xy)
        xy=list(xy)
        x,y=zip(*xy)
        x=np.array(x)
        y=np.array(y)
        return x,y



# data=[]
# SLIDE=64
# T=2048
# for i in range(1,1024):
#   #sample=np.sin(np.linspace(np.pi*np.random.normal(), np.pi*4, np.random.randint(64,1024)))[:64].reshape(64,1)
#   sample=np.sin(np.linspace(np.pi*np.random.rand()*2, np.pi*20+np.pi*np.random.rand()*2, T+SLIDE)).reshape(T+SLIDE,1)
#   data.append(sample)
# data=np.array(data)

# data=np.unique(data,axis=0)
# data=data.transpose((1,0,2))
# # data+=np.random.normal(size=1024*data.shape[0]).reshape(data.shape[0],1024,1)
# train_data_X=data[:int(T*0.5)].copy()
# train_data_X+=np.random.normal(size=train_data_X.shape[1]*train_data_X.shape[0]).reshape(train_data_X.shape[0],train_data_X.shape[1],1)*0.3
# train_data_X=train_data_X.transpose((1,0,2))
# train_data_Y=data[SLIDE:int(T*0.5)+SLIDE].copy()
# train_data_Y=train_data_Y.transpose((1,0,2))
# test_data_X=data[int(T*0.5):len(data)-SLIDE].copy()
# test_data_X+=np.random.normal(size=test_data_X.shape[1]*test_data_X.shape[0]).reshape(test_data_X.shape[0],test_data_X.shape[1],1)*0.3
# test_data_X=test_data_X.transpose((1,0,2))
# test_data_Y=data[int(T*0.5)+SLIDE:].copy()
# test_data_Y=test_data_Y.transpose((1,0,2))
# if  torch.cuda.is_available():
#   train_data_X=np.asarray(train_data_X)
#   test_data_X=np.asarray(test_data_X)