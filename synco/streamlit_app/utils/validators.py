from pathlib import Path
from typing import Tuple, List, Optional


def validate_directory(path: str, dir_type: str) -> Tuple[bool, str]:
    """
    Validate that a directory exists and is accessible.

    Returns: (is_valid, error_message)
    """
    if not path or path.strip() == "":
        return False, f"{dir_type} path cannot be empty"

    path_obj = Path(path).expanduser().resolve()

    if not path_obj.exists():
        return False, f"{dir_type} does not exist: {path_obj}"

    if not path_obj.is_dir():
        return False, f"{dir_type} is not a directory: {path_obj}"

    return True, ""


def validate_paths_config(config: dict) -> Tuple[bool, List[str]]:
    """
    Validate all paths in the configuration.

    Returns: (is_valid, list_of_errors)
    """
    errors = []
    paths = config.get('paths', {})

    # Check required paths
    required_paths = {
        'base': 'Base directory',
        'pipeline_runs': 'Pipeline runs directory',
        'input': 'Input data directory',
    }

    for key, display_name in required_paths.items():
        if key not in paths or not paths[key]:
            errors.append(f"{display_name} is required")
        else:
            is_valid, error_msg = validate_directory(paths[key], display_name)
            if not is_valid:
                errors.append(error_msg)

    # Check optional output path (if provided)
    if 'output' in paths and paths['output']:
        is_valid, error_msg = validate_directory(paths['output'], 'Output directory')
        if not is_valid:
            errors.append(f"Output directory warning: {error_msg}")

    return len(errors) == 0, errors


def validate_cell_lines(cell_lines: list) -> Tuple[bool, str]:
    """
    Validate cell lines list.

    Returns: (is_valid, error_message)
    """
    if not cell_lines or len(cell_lines) == 0:
        return False, "At least one cell line must be specified"

    if not isinstance(cell_lines, list):
        return False, "Cell lines must be a list"

    # Check that all items are non-empty strings
    for cl in cell_lines:
        if not isinstance(cl, str) or not cl.strip():
            return False, "All cell lines must be non-empty strings"

    return True, ""


def validate_prediction_method(method: str) -> Tuple[bool, str]:
    """
    Validate prediction method.

    Returns: (is_valid, error_message)
    """
    valid_methods = ['DrugLogics', 'BooLEVARD']

    if method not in valid_methods:
        return False, f"Prediction method must be one of: {', '.join(valid_methods)}"

    return True, ""


def validate_threshold(threshold: float) -> Tuple[bool, str]:
    """
    Validate synergy threshold.

    Returns: (is_valid, error_message)
    """
    if threshold is None:
        return False, "Threshold cannot be empty"

    try:
        t = float(threshold)
        if t < -1 or t > 1:
            return False, "Threshold must be between -1 and 1"
        return True, ""
    except (TypeError, ValueError):
        return False, "Threshold must be a number"


def validate_config(config: dict) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of the entire configuration.

    Returns: (is_valid, list_of_errors)
    """
    errors = []

    # Validate paths
    paths_valid, path_errors = validate_paths_config(config)
    errors.extend(path_errors)

    # Validate cell lines
    cell_lines = config.get('general', {}).get('cell_lines', [])
    if cell_lines:
        cl_valid, cl_error = validate_cell_lines(cell_lines)
        if not cl_valid:
            errors.append(cl_error)

    # Validate prediction method
    method = config.get('compare', {}).get('prediction_method', 'DrugLogics')
    method_valid, method_error = validate_prediction_method(method)
    if not method_valid:
        errors.append(method_error)

    # Validate threshold
    threshold = config.get('compare', {}).get('threshold')
    threshold_valid, threshold_error = validate_threshold(threshold)
    if not threshold_valid:
        errors.append(threshold_error)

    return len(errors) == 0, errors
