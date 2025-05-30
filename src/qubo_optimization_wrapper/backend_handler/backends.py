from abc import ABC, abstractmethod
from neal import SimulatedAnnealingSampler
from typing import Set, Dict, Any
from qubo_optimization_wrapper.backend_handler.utils import (
    separate_params,
    issue_parameter_filtering_warning,
)

# from dwave.system import DWaveSampler, EmbeddingComposite
import dimod


class Backend(ABC):
    """
    Abstract Base Class for backend implementations.

    Defines a common interface for interacting with different computational
    backends (e.g., simulators, QPUs) for solving QUBO problems.
    It also provides utility methods like parameter filtering.

    """

    def get_allowed_sampler_params(self) -> dict:
        """
        Retrieves the set of allowed parameter names from the backend's sampler.

        :return: A set of allowed parameter names.
        :raises AttributeError: If the sampler or its 'parameters' attribute is not
                                properly initialized or accessible.
        """
        if not hasattr(self, "_sampler") or self._sampler is None:
            raise AttributeError(
                "Sampler not initialized in the backend instance. Cannot retrieve allowed parameters."
            )
        if not hasattr(self._sampler, "parameters") or self._sampler.parameters is None:
            raise AttributeError(
                f"Sampler '{type(self._sampler).__name__}' does not expose its "
                "'parameters' or it is None."
            )
        try:
            return self._sampler.parameters
        except AttributeError:
            raise AttributeError(
                f"Sampler '{type(self._sampler).__name__}' 'parameters' attribute "
                "does not support .keys() or is not dictionary-like."
            )

    def _filter_sampler_params(self, **params_to_filter) -> Dict[str, Any]:
        """
        Filters the given parameters against the allowed parameters of the backend's sampler.
        Issues a warning if any parameters are filtered out.

        :param params_to_filter: Keyword arguments to be filtered.
        :return: A dictionary containing only the parameters recognized by the sampler.
        :raises AttributeError: If the backend's sampler or its parameters attribute
                                is not properly set up or is missing.
        """
        allowed_params_set = set(self.get_allowed_sampler_params().keys())

        filtered_params, discarded_params = separate_params(
            params_to_filter, allowed_params_set
        )

        issue_parameter_filtering_warning(
            sampler_name=type(self._sampler).__name__,
            discarded_params=discarded_params,
            allowed_params_set=allowed_params_set,
            final_filtered_params=filtered_params,
            calling_stacklevel=3,
        )

        return filtered_params

    @abstractmethod
    def sample(self, bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
        """
        Abstract method to sample a given Binary Quadratic Model (BQM).

        Subclasses must implement this method to interact with their specific sampler.

        :param bqm: The binary quadratic model to be sampled.
        :return: A dimod.SampleSet containing the results.
        """
        pass

    @abstractmethod
    def get_properties(self) -> dict:
        """
        Abstract method to retrieve properties of the backend's sampler.

        Subclasses must implement this method.

        :return: A dictionary containing backend-specific properties.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Abstract method to retrieve configurable parameters of the backend's sampler.

        Subclasses must implement this method.

        :return: A dictionary of parameters accepted by the sampler.
        """
        pass


class SimulatedBackend(Backend):
    """
    A backend implementation using a simulated annealing sampler.

    This backend utilizes the `neal.SimulatedAnnealingSampler` to find
    low-energy states for QUBO problems.

    """

    def __init__(self, **config_params):
        """
        Initializes the SimulatedBackend.

        The `SimulatedAnnealingSampler` is instantiated directly. Configuration
        passed via `config_kwargs` is currently logged but not directly used
        to configure the neal sampler at initialization (as neal's sampler is
        primarily configured via its `sample` method parameters).

        :param config_kwargs: Keyword arguments for potential future configuration
                              or for storing backend-specific settings.

        """

        self._sampler = SimulatedAnnealingSampler()
        self._config_params = config_params
        print(f"SimulatedBackend inicializado con config: {self._config_params}")

    def sample(self, bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
        """
        Samples the given BQM using the SimulatedAnnealingSampler.

        Parameters:
            bqm: The binary quadratic model to be sampled
            beta_range (tuple, optional) : A 2-tuple defining the beginning and end of the beta schedule, where beta is the inverse temperature. The schedule is interpolated within this range according to the value specified by beta_schedule_type. Default range is set based on the total bias associated with each node.
            num_reads (int, optional, default=len(initial_states) or 1) : Number of reads. Each read is generated by one run of the simulated annealing algorithm. If num_reads is not explicitly given, it is selected to match the number of initial states given. If initial states are not provided, only one read is performed.
            num_sweeps (int, optional, default=1000) : Number of sweeps or steps.
            beta_schedule_type (string, optional, default='geometric') : Beta schedule type, or how the beta values are interpolated between the given ‘beta_range’. Supported values are: linear, geometric.
            seed (int, optional) : Seed to use for the PRNG. Specifying a particular seed with a constant set of parameters produces identical results. If not provided, a random seed is chosen.
            interrupt_function (function, optional) : If provided, interrupt_function is called with no parameters between each sample of simulated annealing. If the function returns True, then simulated annealing will terminate and return with all of the samples and energies found so far.
            initial_states (samples-like, optional, default=None) : One or more samples, each defining an initial state for all the problem variables. Initial states are given one per read, but if fewer than num_reads initial states are defined, additional values are generated as specified by initial_states_generator. See func:.as_samples for a description of “samples-like”.
            initial_states_generator (str, 'none'/'tile'/'random', optional, default='random') : Defines the expansion of initial_states if fewer than num_reads are specified: ”none”:If the number of initial states specified is smaller than num_reads, raises ValueError. ”tile”:Reuses the specified initial states if fewer than num_reads or truncates if greater. ”random”:Expands the specified initial states with randomly generated states if fewer than num_reads or truncates if greater.

        :return: A dimod.SampleSet containing the samples from the simulated annealing process.

        """

        filtered_params = self._filter_sampler_params(**self._config_params)
        return self._sampler.sample(bqm, **filtered_params)

    def get_properties(self) -> dict:
        """
        Retrieves properties of the SimulatedAnnealingSampler.

        These properties provide additional information about the sampler's state or capabilities.

        :return: A dict containing properties of the `neal.SimulatedAnnealingSampler`.

        """

        return self._sampler.properties

    def get_parameters(self) -> dict:
        """
        Retrieves the parameters accepted by the SimulatedAnnealingSampler's sample method.

        The keys are the keyword parameters (allowed kwargs) and values are lists of
        SimulatedAnnealingSampler.properties relevant to each parameter.

        :return: A dict of parameters accepted by `neal.SimulatedAnnealingSampler.sample()`.

        """

        return self._sampler.parameters


class QPUBackend(Backend):
    """
    A backend implementation for interacting with D-Wave Quantum Processing Units (QPUs).

    This backend uses DWaveSampler wrapped with EmbeddingComposite to handle minor-embedding

    and provide a sampling interface for QUBO problems.

    """

    def __init__(self, **config_params):
        """
        Initializes the QPUBackend and the underlying D-Wave sampler.

        Configuration for the DWaveSampler (e.g., 'solver_name', 'token', 'endpoint')
        should be passed as keyword arguments.

        :param config_kwargs: Keyword arguments to configure the DWaveSampler.
                              Common arguments include 'solver' (or 'solver_name'), 'token', 'endpoint'.
        :raises RuntimeError: If the DWaveSampler cannot be initialized with the
                              provided configuration (e.g., solver not found, authentication issues).

        """

        try:
            self._sampler = None  # EmbeddingComposite(DWaveSampler(**config_kwargs))
            self._config_params = config_params
            solver_used = config_params.get(
                "solver", config_params.get("solver_name", "default/unknown")
            )
            print(
                f"QPUBackend initialized for solver: {solver_used} with config: {config_params}"
            )

        except Exception as e:  # Ser más específico con la excepción si es posible (e.g., dwave.cloud.exceptions.SolverNotFoundError)
            solver_info = config_params.get(
                "solver_name", config_params.get("solver", str(config_params))
            )
            raise RuntimeError(
                f"No se pudo inicializar DWaveSampler para '{solver_info}': {e}"
            ) from e

    def sample(self, bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
        """
        Samples the given Binary Quadratic Model (BQM) using the configured D-Wave QPU.

        This method utilizes `EmbeddingComposite` to handle minor-embedding.
        Only parameters recognized by the underlying `DWaveSampler` will be passed to it.

        :param bqm: The binary quadratic model to be sampled.
        :param params: Additional keyword arguments to pass to the `DWaveSampler`'s
                       sample method (e.g., 'num_reads', 'annealing_time', 'label').
        :return: A `dimod.SampleSet` containing the results from the QPU.
        :raises RuntimeError: If the internal sampler is not initialized.
        :raises AttributeError: If parameter filtering fails due to sampler misconfiguration.
        """
        if not self._sampler:
            raise RuntimeError("QPU Sampler is not initialized.")

        filtered_params = self._filter_sampler_params(**self._config_params)
        return self._sampler.sample(bqm, **filtered_params)

    def get_properties(self) -> dict:
        """
        Retrieves properties of the underlying D-Wave QPU sampler.

        These properties typically include information about the QPU.

        :return: A dictionary of properties from the `DWaveSampler`, or an empty
                 dict if the sampler is not available or has no properties.
        """

        return getattr(self._sampler, "properties", {})

    def get_parameters(self) -> dict:
        """
        Retrieves configurable parameters of the underlying D-Wave QPU sampler.

        These parameters are the keyword arguments accepted by the `DWaveSampler`'s
        sampling methods.

        :return: A dictionary of parameters from the `DWaveSampler`, or an empty
                 dict if the sampler is not available or has no parameters.
        """

        return getattr(self._sampler, "parameters", {})
