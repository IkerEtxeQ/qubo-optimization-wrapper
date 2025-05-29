from .backend_factory import BackendFactory
from qubo_optimization_wrapper.backend_handler.backends.backend_interface import Backend
from qubo_optimization_wrapper.hamiltonian_creation.hamiltonian import Hamiltonian
import dimod

# TODO: QUITAR HAMILTONIANO DE LOS ARGUMENTOS DEL OBJETO, NO DEBE DEPENDER DE EL.


class ExecutionBackend:
    def __init__(self, hamiltonian: Hamiltonian, backend_type: str):
        self._hamiltonian = hamiltonian
        self._backend_factory = BackendFactory()
        self._current_backend: Backend = None
        self._current_backend_execution_info = None
        self._decoded_sampleset = None
        self.set_backend(backend_type)

    def set_backend(self, backend_type: str):
        self._current_backend = self._backend_factory.create_backend(backend_type)
        self._current_backend_execution_info = self._current_backend.sample.__doc__
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
        print(f"{self._current_backend_execution_info}")

    def get_decoded_sampleset(self) -> dimod.SampleSet:
        if not self._decoded_sampleset:
            return print("Job was not executed in selected backend yet")
        return self._decoded_sampleset

    def submit_job(self, **params) -> dimod.SampleSet:
        """Sample from a Hamiltonian simbolic expresion and a coef dict named lambda dict.

        Args:
            H: Hamiltonian simbolic expresion.
            lambda_dict (dict) : Coefficient dictionary.

        """

        if not self._current_backend:
            raise RuntimeError("No backend configured. Call set_backend() first.")

        bqm = self._hamiltonian.get_compiled_hamiltonian().get_bqm()

        sampleset = self._current_backend.sample(bqm, **params)

        self._decoded_sampleset = self._decode_sampleset(sampleset)

    def _decode_sampleset(self, sampleset):
        """More info: https://test-projecttemplate-dimod.readthedocs.io/en/latest/reference/sampleset.html#id1

        Returns:
            de
        """
        model = self._hamiltonian.get_compiled_hamiltonian().get_model()
        lambda_dict = self._hamiltonian.get_lambda_dict()

        if lambda_dict:
            decoded_samples = model.decode_sampleset(sampleset, feed_dict=lambda_dict)
        else:
            decoded_samples = model.decode_sampleset(sampleset)

        return decoded_samples
