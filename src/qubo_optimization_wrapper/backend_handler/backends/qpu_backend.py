from qubo_optimization_wrapper.backend_handler.backends.backend_interface import Backend
import dimod
# from dwave.system import DWaveSampler, EmbeddingComposite


class QPUBackend(Backend):
    def __init__(self):
        self._sampler = None  # EmbeddingComposite(DWaveSampler())

    def sample(self, bqm: dimod.BinaryQuadraticModel, **params) -> dimod.SampleSet:
        return self._sampler.sample(bqm, **params)

    def get_properties(self) -> dict:
        return getattr(self._sampler, "properties", {})

    def get_parameters(self) -> dict:
        return getattr(self._sampler, "parameters", {})
