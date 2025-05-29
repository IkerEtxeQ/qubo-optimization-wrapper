from abc import ABC, abstractmethod
import dimod


class Backend(ABC):
    def __init__(self):
        self._sampler = None

    @abstractmethod
    def sample(self, bqm: dimod.BinaryQuadraticModel, **params) -> dimod.SampleSet:
        pass

    @abstractmethod
    def get_properties(self) -> dict:
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        pass
