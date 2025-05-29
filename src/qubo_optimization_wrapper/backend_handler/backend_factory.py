from .backends.qpu_backend import QPUBackend
from .backends.simulated_backend import SimulatedBackend
from qubo_optimization_wrapper.backend_handler.backends.backend_interface import Backend


class BackendFactory:
    def __init__(self):
        pass

    def create_backend(self, backend_type: str) -> Backend:
        if backend_type == "dwave_qpu":
            return QPUBackend()
        elif backend_type == "dwave_hybrid":
            return QPUBackend()
        elif backend_type == "simulated_annealing":
            return SimulatedBackend()
        else:
            raise ValueError(f"Backend no soportado: {backend_type}")
