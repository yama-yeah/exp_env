from abc import ABCMeta, abstractmethod
from torch import Tensor

class BaseLinearModel(metaclass=ABCMeta):
    # 逆行列から重みを求める
    # 局所最適解に陥りやすい
    @abstractmethod
    def fit(self, X, y)->None:
        pass

    @abstractmethod
    def loss(self, X, y)->Tensor:
        pass
            
        