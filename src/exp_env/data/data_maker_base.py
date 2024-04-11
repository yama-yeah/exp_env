from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

class DataMakerSpec:
    pass

class DataMaker(ABC):
    def __init__(self, spec: DataMakerSpec):
        self.spec = spec
    @abstractmethod
    def make(self, *args, **kwargs):
        pass

class CycleDataSliderSpec(DataMakerSpec):
    cycle_T: int
    slide: int
    def __init__(self, cycle_T: int, slide: int,how_many:Optional[int]=1,length:Optional[int]=None):
        self.cycle_T = cycle_T
        self.slide = slide
        if how_many is not None:
            self.length = cycle_T*how_many
        elif length is not None:
            self.length = length
        else:
            raise ValueError("Either how_many or length must be specified")

class CycleDataSlider(DataMaker):
    def __init__(self, spec: CycleDataSliderSpec):
        self.spec = spec
    @abstractmethod
    def make(self, *args, **kwargs):
        pass
    
