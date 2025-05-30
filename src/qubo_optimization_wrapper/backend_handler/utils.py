# qubo_optimization_wrapper/backend_handler/utils.py
import warnings
from typing import Set, Dict, Any, Tuple  # Corregido Tuple


def separate_params(
    params_to_process: Dict[str, Any], allowed_params_set: Set[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # Usar Tuple de typing
    """
    Separates provided parameters into filtered (allowed) and discarded (not allowed) ones.

    :param params_to_process: The dictionary of parameters to separate.
    :param allowed_params_set: A set of allowed parameter names.
    :return: A tuple containing two dictionaries: (filtered_params, discarded_params).
    """
    filtered_params = {}
    discarded_params = {}
    for key, value in params_to_process.items():
        if key in allowed_params_set:
            filtered_params[key] = value
        else:
            discarded_params[key] = value
    return filtered_params, discarded_params


def issue_parameter_filtering_warning(
    sampler_name: str,
    discarded_params: Dict[str, Any],
    allowed_params_set: Set[str],
    final_filtered_params: Dict[str, Any],
    calling_stacklevel: int = 2,  # Nivel de stack para que la advertencia apunte correctamente
):
    """
    Formats and issues a UserWarning if any parameters were discarded.

    :param sampler_name: Name of the sampler for the warning message.
    :param discarded_params: Dictionary of parameters that were filtered out.
    :param allowed_params_set: Set of allowed parameter names for the sampler.
    :param final_filtered_params: Dictionary of parameters that will be used.
    :param calling_stacklevel: Adjusts where the warning message points in the call stack.
    """
    if not discarded_params:
        return  # No warning needed

    discarded_keys_str = ", ".join(f"'{k}'" for k in sorted(discarded_params.keys()))

    recognized_keys_list_str = "\n    - ".join(sorted(list(allowed_params_set)))
    if recognized_keys_list_str:
        recognized_keys_list_str = "- " + recognized_keys_list_str
    else:
        recognized_keys_list_str = "(No recognized parameters found for this sampler)"

    warning_message_parts = [
        f"Warning: Parameter Filtering Report for '{sampler_name}' sampler:",
        f"  The following parameters were provided but are not recognized by the sampler and have been filtered out:",
        f"    {discarded_keys_str}",
        f"  Recognized parameters by '{sampler_name}' are:",
        f"    {recognized_keys_list_str}",
        f"  Final effective configuration parameters passed to the sampler:",
        f"    {final_filtered_params}",
    ]
    warning_message = "\n".join(warning_message_parts)

    warnings.warn(warning_message, UserWarning, stacklevel=calling_stacklevel)
