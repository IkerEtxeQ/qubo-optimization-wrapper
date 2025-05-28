from qubo_optimization_wrapper.backend_handler.backends.backend_interface import Backend
import dimod


class DWaveHardwareBackend(Backend):
    def __init__(
        self,
        backend_config,
        client_type="qpu",
        solver_name_or_criteria=None,
        use_embedding=True,
    ):
        super().__init__(backend_config)
        self._client_type = client_type
        self._solver_name_or_criteria = solver_name_or_criteria
        self._use_embedding = use_embedding

        try:
            if self.client_type == "qpu":
                pass
            elif self.client_type == "hybrid":
                pass
            else:
                raise ValueError(
                    f"Tipo de cliente D-Wave no soportado: {self.client_type}"
                )
        except Exception as e:
            raise ConnectionError(
                f"No se pudo inicializar DWaveSampler/LeapHybridSampler: {e}"
            )

    def sample(self, bqm: dimod.BinaryQuadraticModel, **params) -> dimod.SampleSet:
        return self._sampler.sample(bqm, **params)

    def get_properties(self) -> dict:
        return getattr(self._sampler, "properties", {})

    def get_parameters(self) -> dict:
        return getattr(self._sampler, "parameters", {})
