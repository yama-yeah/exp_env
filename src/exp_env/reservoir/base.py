from abc import ABCMeta, abstractmethod
from torch import Tensor

class BaseReservoirModule(metaclass=ABCMeta):
    @abstractmethod
    def next(self, x:Tensor)->Tensor:
        pass
    @abstractmethod
    def reset(self)->None:
        pass
    @abstractmethod
    def add_log_state(self, state:Tensor)->None:
        pass