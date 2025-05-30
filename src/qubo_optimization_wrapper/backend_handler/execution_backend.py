from qubo_optimization_wrapper.backend_handler.backend_factory import create_backend
from qubo_optimization_wrapper.backend_handler.backends import Backend
from qubo_optimization_wrapper.hamiltonian_creation.hamiltonian import Hamiltonian
import dimod


class ExecutionBackend:
    """
    Manages the execution of QUBO problems on different computational backends.

    This class provides an interface to set a specific backend (e.g., simulator, QPU),
    submit optimization jobs, and retrieve results and backend information.

    """

    def __init__(self, backend_type: str, **backend_config):
        """
        Initializes the ExecutionBackend with a specified backend.

        :param backend_type: The type identifier string for the backend (e.g., "simulated_annealing").
        :param backend_config: Keyword arguments for backend-specific configuration.

        """

        self._current_backend: Backend = None
        self._sampleset: dimod.SampleSet = None
        self._backend_config = backend_config
        self.set_current_backend(backend_type, **backend_config)

    @property
    def current_backend(self) -> Backend:
        """
        Backend: The currently active computational backend instance.

        :raises RuntimeError: If the backend has not been initialized.

        """

        if not self.current_backend:
            raise RuntimeError("Backend no ha sido inicializado correctamente.")
        return self.current_backend

    def set_current_backend(self, backend_type: str, **backend_config):
        """
        Sets or changes the active computational backend.

        :param backend_type: The type identifier string for the backend.
        :param backend_config: Keyword arguments for backend-specific configuration.

        """

        self._current_backend = create_backend(backend_type, **backend_config)
        print(f"Backend cambiado a: {backend_type}")

    @property
    def sampleset(self) -> dimod.SampleSet:
        """
        dimod.SampleSet: The result of the last executed job.

        :raises AttributeError: If accessed before a job has been successfully submitted and processed.
        """
        if self._sampleset is None:
            raise AttributeError(
                "No sampleset available. A job must be submitted and processed first using 'submit_job()'."
            )
        return self._sampleset

    def _ensure_backend_is_set(self):
        """
        Internal helper to ensure a backend is configured before an operation.

        :raises RuntimeError: If no backend is currently configured.

        """

        if not self._current_backend:
            raise RuntimeError(
                "No backend configurado. Llama a set_current_backend() o inicializa la clase correctamente."
            )

    def get_backend_properties(self) -> dict:
        """
        Retrieves properties of the current backend.

        :return: A dictionary containing backend-specific properties.
        :raises RuntimeError: If no backend is configured.

        """

        self._ensure_backend_is_set()
        return self._current_backend.get_properties()

    def get_backend_parameters(self) -> dict:
        """
        Retrieves configurable parameters of the current backend.

        :return: A dictionary of parameters accepted by the backend's sampler.
        :raises RuntimeError: If no backend is configured.

        """

        self._ensure_backend_is_set()
        return self._current_backend.get_allowed_sampler_params()

    def print_current_backend_execution_info(self):
        """
        Prints documentation or execution information for the current backend's sample method.

        :raises RuntimeError: If no backend is configured.

        """

        self._ensure_backend_is_set()
        docstring = getattr(
            getattr(self._current_backend, "sample", None), "__doc__", None
        )
        if docstring:
            return print(str(docstring))
        return print(
            "No hay información de documentación disponible para el método sample del backend actual."
        )

    def submit_job(self, hamiltonian: Hamiltonian):
        """
        Submits a Hamiltonian to the current backend for sampling.

        :param hamiltonian: The Hamiltonian object representing the problem.
        :param params: Additional keyword arguments to pass to the backend's sample method
                       (e.g., num_reads, annealing_time).
        :return: A dimod.SampleSet containing the results from the backend.
        :raises RuntimeError: If no backend is configured.

        """

        self._ensure_backend_is_set()

        bqm = hamiltonian.get_compiled_hamiltonian().get_bqm()

        self._sampleset = self._current_backend.sample(bqm)

        return self._sampleset
