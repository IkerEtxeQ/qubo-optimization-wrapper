from typing import Type, Dict, Any
from qubo_optimization_wrapper.backend_handler.backends import (
    Backend,
    QPUBackend,
    SimulatedBackend,
)


_BACKEND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dwave_qpu": {
        "class": QPUBackend,
        "default_config": {"solver_name": "Advantage_system6.1"},
    },
    "dwave_hybrid": {  # Si QPUBackend lo maneja con config
        "class": QPUBackend,
        "default_config": {"solver_name": "hybrid_binary_quadratic_model_version2"},
    },
    "simulated_annealing": {
        "class": SimulatedBackend,
        "default_config": {"num_reads": 10},
    },
}


def create_backend(backend_type: str, **override_config_kwargs) -> Backend:
    """
    Crea una instancia de backend basada en el backend_type.
    """
    entry = _BACKEND_REGISTRY.get(backend_type)
    if not entry:
        raise ValueError(
            f"Backend no soportado: {backend_type}. "
            f"Opciones disponibles: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_class: Type[Backend] = entry["class"]

    final_config = entry.get("default_config", {}).copy()
    final_config.update(override_config_kwargs)

    return backend_class(**final_config)
