from .backends.dwave_hardware_backend import DWaveHardwareBackend
from .backends.simulated_backend import SimulatedBackend
from qubo_optimization_wrapper.backend_handler.backends.backend_interface import Backend


class BackendFactory:
    def __init__(self):
        pass

    def create_backend(self, backend_type: str, backend_config) -> Backend:
        if backend_type == "dwave_qpu":
            return DWaveHardwareBackend(
                backend_config,
                client_type="qpu",
            )
        elif backend_type == "dwave_hybrid":
            return DWaveHardwareBackend(backend_config, client_type="hybrid")
        elif backend_type == "simulated_annealing":
            return SimulatedBackend()
        else:
            raise ValueError(f"Backend no soportado: {backend_type}")
