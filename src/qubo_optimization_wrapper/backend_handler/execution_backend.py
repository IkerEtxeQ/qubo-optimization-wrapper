from .backend_factory import BackendFactory
from qubo_optimization_wrapper.backend_handler.backends.backend_interface import Backend
from qubo_optimization_wrapper.hamiltonian_creation.hamiltonian import Hamiltonian
import dimod


class ExecutionBackend:
    def __init__(self, backend_type: str):
        self._backend_factory = BackendFactory()
        self._current_backend: Backend = None
        self._sampleset = None
        self.set_backend(backend_type)

    def set_backend(self, backend_type: str):
        self._current_backend = self._backend_factory.create_backend(backend_type)
        print(f"Backend cambiado a: {backend_type}")

    def get_backend_properties(self) -> dict:
        if not self._current_backend:
            return {}
        return self._current_backend.get_properties()

    def get_backend_parameters(self) -> dict:
        if not self._current_backend:
            return {}
        return self._current_backend.get_parameters()

    def get_current_backend_execution_info(self):
        print(f"{self._current_backend.sample.__doc__}")

    def get_sampleset(self) -> dimod.SampleSet:
        if not self._sampleset:
            return print("Job was not executed in selected backend yet")
        return self._sampleset

    def submit_job(self, hamiltonian: Hamiltonian, **params):
        """Sample from a Hamiltonian simbolic expresion and a coef dict named lambda dict.

        Args:
            H: Hamiltonian simbolic expresion.

        """

        if not self._current_backend:
            raise RuntimeError("No backend configured. Call set_backend() first.")

        bqm = hamiltonian.get_compiled_hamiltonian().get_bqm()

        self._sampleset = self._current_backend.sample(bqm, **params).aggregate()
