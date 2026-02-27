# synco/main.py

from copy import deepcopy
from typing import Optional
from pathlib import Path

from .config import BASE_DEFAULTS
from .utils import deep_merge, load_dataframe, echo_message, save_file
from .features import (
    DataLoader,
    resolve_cell_lines,
    get_drugprofiles,
    get_synergy_predictions,
    converge_synergies,
    compare_synergies,
    calculate_roc_metrics
)



#///////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN SYNCO PIPELINE 
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////
def build_pipeline_config(user_config: dict) -> dict:
    """
    Build the final pipeline configuration by merging user-provided settings with base defaults.
    Includes validation of required configuration fields.

    Args:
        user_config (dict): User-provided configuration settings.

    Returns:
        dict: The final merged configuration.
        
    Raises:
        ValueError: If required configuration fields are missing or invalid.
    """
    # Validate required configuration structure
    if "paths" not in user_config:
        raise ValueError("Configuration must include 'paths' section")
    
    paths = user_config["paths"]
    
    # Validate required paths
    required_paths = ["base", "pipeline_runs", "input"]
    for path_key in required_paths:
        if path_key not in paths:
            raise ValueError(f"Required path '{path_key}' is missing from configuration")
        if paths[path_key] is None:
            raise ValueError(f"Required path '{path_key}' cannot be None")
    
    # Validate general section
    if "general" not in user_config:
        raise ValueError("Configuration must include 'general' section")
        
    general = user_config["general"]
    if "cell_lines" not in general:
        raise ValueError("Configuration must include 'cell_lines' in general section")
    
    # cell_lines can be: list (manual or empty for auto), string (CSV file), or None
    cell_lines_input = general["cell_lines"]
    if cell_lines_input is not None:
        if not isinstance(cell_lines_input, (list, str)):
            raise ValueError("'cell_lines' must be a list, string (CSV filename), or None")

    # If the user already passed a fully-built pipeline configuration (contains 'steps'),
    # assume it's merged and return it unchanged. This avoids overwriting previously
    # merged `steps` when notebook code builds the config first and then calls
    # `run_pipeline(pipeline_config, ...)` with that merged dict.
    if isinstance(user_config, dict) and "steps" in user_config and isinstance(user_config["steps"], dict):
        return user_config

    # Start with user config and merge base defaults
    final_config = {
        "paths": user_config.get("paths", {}),
        "general": user_config.get("general", {}),
        "compare": user_config.get("compare", {}),
        "output_control": user_config.get("output_control", {}),
        "steps": deepcopy(BASE_DEFAULTS),
    }

    # Merge in any "advance" overrides from user
    advance = user_config.get("advance", {})
    deep_merge(final_config["steps"], advance)

    return final_config

#/////////////////////////////////////////////////////
def _validate_stop_after(stop_after: Optional[str]) -> None:
    """
    Validate that the stop_after parameter is a valid pipeline step.
    
    Args:
        stop_after: The stop_after parameter to validate.
        
    Raises:
        ValueError: If stop_after is not a valid step.
    """
    if stop_after is None:
        return
        
    valid_stops = [
        "fetch", 
        "drug_profiles", 
        "synergy_predictions", 
        "synergy_convergence", 
        "synergy_comparison",
        "roc_metrics"
    ]
    
    if stop_after not in valid_stops:
        raise ValueError(f"Invalid stop_after value '{stop_after}'. Valid options are: {valid_stops}")

#/////////////////////////////////////////////////////
def run_pipeline(
        config: dict,
        plan: bool = False,
        stop_after: Optional[str] = None,
        verbose: bool = False
    ):
    """
    Run the SYNCO pipeline with the given configuration.
    STEPS:
    1. setup & fetch
    2. drug profiles
    3. synergy predictions
    4. converge synergies
    5. compare synergies
    6. calculate ROC metrics

    Args:
        config (dict): The configuration dictionary for the pipeline.
        plan (bool): If True, only plan the pipeline without executing.
        stop_after (Optional[str]): If specified, stop after this step.
            Valid options: 'fetch', 'drug_profiles', 'synergy_predictions', 
            'synergy_convergence', 'synergy_comparison'
        verbose (bool): If True, print detailed information during execution.
    """
    # ------------------------------------------------------
    # STEP 0: CONFIG
    # ------------------------------------------------------
    
    # Validate stop_after parameter
    _validate_stop_after(stop_after)
    
    # Config loading
    cfg = build_pipeline_config(config)
    paths = cfg["paths"]
    general = cfg["general"]
    compare = cfg["compare"]
    steps = cfg["steps"]
    output_control = cfg.get("output_control", {})
    use_output_control = bool(output_control.get("enabled", False))
    threshold = float(cfg["compare"].get("threshold", 0.0))
    threshold_offsets = cfg["compare"].get("threshold_offsets")
    roc_bootstrap_n = cfg["compare"].get("roc_bootstrap_n")
    roc_bootstrap_n = int(roc_bootstrap_n) if roc_bootstrap_n is not None else None
    roc_bootstrap_ci = float(cfg["compare"].get("roc_bootstrap_ci", 0.95))

    # ------------------------------------------------------
    # Common paths normalization
    base = Path(paths.get("base", ".")).expanduser().resolve()
    input = Path(paths.get("input", base / "input")).expanduser()
    
    # Handle output path - allow None for no output
    output_path_config = paths.get("output")
    if output_path_config is not None:
        output = Path(output_path_config).expanduser()
    else:
        output = None

    shared_output = None
    if use_output_control:
        shared_output_config = output_control.get("shared_output")
        if shared_output_config:
            shared_output = Path(shared_output_config).expanduser()
    
    runs = Path(paths["pipeline_runs"]).expanduser()

    # Resolve cell lines (manual list, CSV file, or auto-discover)
    cell_lines = resolve_cell_lines(
        cell_line_input=general.get("cell_lines"),
        pipeline_runs_path=str(runs),
        input_path=str(input),
        verbose=verbose
    )
    if verbose:
        echo_message(f"Resolved {len(cell_lines)} cell lines: {cell_lines}", verbose)

    # Ensure directories exist
    directories_to_create = [input, runs]
    if output is not None:
        directories_to_create.append(output)
    if shared_output is not None:
        directories_to_create.append(shared_output)
        
    for p in directories_to_create:
        p.mkdir(parents=True, exist_ok=True)
        echo_message(f"Ensured directory: {p}", verbose)

    # ------------------------------------------------------
    # PLAN - pipeline execution steps
    if plan:
        print("Pipeline plan:\n")
        print("Directories:")
        print(f" - Input: {input}")
        print(f" - Output: {output}")
        print(f" - Synergy prediction Runs: {runs}\n")
        print(f"Pipeline will execute until step: {stop_after}" if stop_after else "Pipeline will execute all steps.")
        print("\nConfig variables:\n")
        print(f"Cell lines: {cell_lines}")
        print(f"Synergy threshold: {threshold}")
        print(f"Prediction method: {compare.get('prediction_method', 'DrugLogics')}")
        print(f"Analysis mode: {compare.get('analysis_mode', 'default')}")
        print("\nSteps configuration:\n")
        for step_name, step_config in steps.items():
            print(f"Step '{step_name}':")
            for param_name, param_value in step_config.items():
                print(f"  - {param_name}: {param_value}")
    artifacts: dict = {"synco_configuration": cfg}

    if not plan:
        echo_message("\nStarting SYNCO Pipeline Execution", verbose)
    # ------------------------------------------------------
    # STEP 1: SETUP & FETCH
    # ------------------------------------------------------
    try:
        # Load experimental synergy results
        dl_cfg = steps.get('data_loading', {})
        synergy_filename = dl_cfg.get('synergy_filename')
        synergy_pattern = dl_cfg.get('synergy_pattern', 'synergies_observed*.csv')

        # Determine where to look for the experimental synergies file(s)
        if synergy_filename:
            # If an absolute path was provided, load directly from its parent folder
            sym_path = Path(synergy_filename)
            if sym_path.is_absolute():
                folder_to_search = sym_path.parent
                pattern_to_use = sym_path.name
            else:
                # Treat as a filename or relative path under the configured input folder
                folder_to_search = input
                pattern_to_use = str(sym_path)

            synergies_exp_processed = load_dataframe(folder=folder_to_search, pattern=pattern_to_use)
        else:
            # Use the configured pattern to find the file inside `input`
            folder_to_search = input
            pattern_to_use = synergy_pattern
            synergies_exp_processed = load_dataframe(folder=folder_to_search, pattern=pattern_to_use)
        echo_message(f"Loaded experimental synergies using pattern/filename: {synergy_filename or synergy_pattern}", verbose)
        
        # Load prediction results
        pipeline_results = DataLoader(
            base_path=base,
            cell_info_path=input,
            cell_line_list=cell_lines,
            prediction_method=compare.get("prediction_method", "DrugLogics"),
            experimental_observations=steps['data_loading'].get("experimental_observations", False),
            run_results_path=runs,
            analysis_folder=steps['data_loading'].get("analysis_folder", None),
            run_date=general.get("run_date", None),
            verbose=verbose,
        )
        if not plan:
            echo_message("\nStarting STEP 1: Setup & Fetch", verbose)
            pipeline_results.make_analysis_folders()
            synergy_data_dict = pipeline_results.fetch_synergy_data(
            experimental_observations=steps['data_loading'].get("experimental_observations", False)
            )
            artifacts["synergy_data_dict"] = synergy_data_dict
            echo_message("\nSynergy data loaded successfully.", verbose)
    except Exception as e:
        raise RuntimeError(f"Error in STEP 1: {e}") from e
    if stop_after == "fetch":
        echo_message("\nStopping pipeline after STEP 1: Fetch", verbose)
        return artifacts

    # ------------------------------------------------------
    # STEP 2: DRUG PROFILES
    # ------------------------------------------------------
    try:
        if not plan:
            echo_message("\nStarting STEP 2: Drug Profiles", verbose)
            profiles_output = output
            if use_output_control:
                profiles_output = shared_output if output_control.get("write_profiles", False) else None
            drug_profiles = get_drugprofiles(
                input_path=input,
                output_path=profiles_output,
                inhibitor=steps['data_loading'].get("inhibitor_profiles", 'inhibitor_profiles'),
                drug_info=steps['data_loading'].get("drug_info", 'drug_profiles')
                )
            artifacts["drug_profiles"] = drug_profiles
            echo_message("\nDrug profiles loaded successfully.", verbose)
    except Exception as e:
        raise RuntimeError(f"Error in STEP 2: {e}") from e
    if stop_after == "drug_profiles":
        echo_message("\nStopping pipeline after STEP 2: Drug Profiles", verbose)
        return artifacts

    # ------------------------------------------------------
    # STEP 3: SYNERGY PREDICTIONS
    # ------------------------------------------------------
    pred_cfg = steps['synergy_predictions']
    try:
        if not plan:
            echo_message("\nStarting STEP 3: Synergy Predictions", verbose)
            predictions_output = output
            if use_output_control:
                predictions_output = output if output_control.get("write_synergy_predictions", False) else None
            synergy_pred_results = get_synergy_predictions(
                synergy_results_dict=synergy_data_dict,
                combination_type=pred_cfg["combination_type"],
                mapping_names_dict=drug_profiles['PD_drugnames_dict'],
                mapping_target_dict=drug_profiles['PD_targets_dict'],
                drugID_type=pred_cfg["drugID_type"],
                target_type=pred_cfg["target_type"],
                remove_duplicates=pred_cfg["remove_duplicates"],
                add_experimental_observations=pred_cfg["add_experimental_observations"],
                output_path=predictions_output
            )
            artifacts["synergy_predictions"] = synergy_pred_results
            echo_message("\nSynergy predictions completed successfully.", verbose)
    except KeyError as e:
        raise RuntimeError(f"Missing key in synergy predictions: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error in STEP 3: {e}") from e
    if stop_after == "synergy_predictions":
        echo_message("\nStopping pipeline after STEP 3: Synergy Predictions", verbose)
        return artifacts
    
    # ------------------------------------------------------
    # STEP 4: SYNERGY CONVERGE
    # ------------------------------------------------------
    conv_cfg = steps['synergy_convergence']
    try:
        if not plan:
            echo_message("\nStarting STEP 4: Synergy Converge", verbose)
            converge_output = output
            if use_output_control:
                converge_output = None
            converge_experimental = converge_synergies(
                df=synergies_exp_processed,
                anchorID=conv_cfg['anchorID'],
                libraryID=conv_cfg['libraryID'],
                anchor_name=conv_cfg['anchor_name'],
                library_name=conv_cfg['library_name'],
                inhibitor_groups=drug_profiles['PD_inhibitors_dict'],
                targets_dict=drug_profiles['PD_targets_dict'],
                cell_line=conv_cfg['cell_line_column'],
                cell_line_list=cell_lines,
                predicted=conv_cfg['predicted_data'],
                output_path=converge_output
            )
            exp_full_df, _, exp_inhibitor_group_synergies_df = converge_experimental
            converge_predictions = converge_synergies(
                df=synergy_pred_results,
                anchorID=conv_cfg['anchorID'],
                libraryID=conv_cfg['libraryID'],
                anchor_name=conv_cfg['anchor_name'],
                library_name=conv_cfg['library_name'],
                inhibitor_groups=drug_profiles['PD_inhibitors_dict'],
                targets_dict=drug_profiles['PD_targets_dict'],
                cell_line=conv_cfg['cell_line_column'],
                cell_line_list=cell_lines,
                predicted=True,
                output_path=converge_output
            )
            pred_full_df, _, pred_inhibitor_group_synergies_df = converge_predictions
            if use_output_control:
                if output_control.get("write_experimental_full_df", False) and shared_output is not None:
                    save_file(exp_full_df, shared_output / "experimental_full_df.csv")
                if output_control.get("write_predictions_full_df", False) and output is not None:
                    save_file(pred_full_df, output / "predictions_full_df.csv")
            artifacts["experimental_convergence"] = converge_experimental
            artifacts["predictions_convergence"] = converge_predictions
            echo_message("\nSynergy convergence completed successfully.", verbose)
    except KeyError as e:
        raise RuntimeError(f"Missing key in synergy convergence: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error in STEP 4: {e}") from e
    if stop_after == "synergy_convergence":
        echo_message("\nStopping pipeline after STEP 4: Synergy Convergence", verbose)
        return artifacts

    # ------------------------------------------------------
    # STEP 5: SYNERGY COMPARISON
    # ------------------------------------------------------
    try:
        if not plan:
            echo_message("\nStarting STEP 5: Synergy Comparison", verbose)
            compare_output = output
            if use_output_control:
                compare_output = output if output_control.get("write_compare_outputs", False) else None
            comparison_results, skipped_info, comparison_summary, pair_details_df = compare_synergies(
                df_experiment=exp_inhibitor_group_synergies_df,
                df_prediction=pred_inhibitor_group_synergies_df,
                synergy_column=compare.get("synergy_column", "synergy"),
                cell_line_list=cell_lines,
                threshold=threshold,
                analysis_mode=compare.get("analysis_mode", "inhibitor_combination"),
                duplicate_strategy=compare.get("duplicate_strategy", "mean"),
                output_path=compare_output,
                debug_items=compare.get("debug_items", None),
                return_pair_details=True,
                df_experiment_full=exp_full_df,
                df_prediction_full=pred_full_df,
            )
            artifacts["synergy_comparison"] = comparison_results
            artifacts["skipped_info"] = skipped_info
            artifacts["comparison_summary"] = comparison_summary
            artifacts["pair_details"] = pair_details_df
            echo_message("\nSynergy comparison completed successfully.", verbose)
    except KeyError as e:
        raise RuntimeError(f"Missing key in synergy comparison: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error in STEP 5: {e}") from e
    if stop_after == "synergy_comparison":
        echo_message("\nStopping pipeline after STEP 5: Synergy Comparison", verbose)
        return artifacts

    # ------------------------------------------------------
    # STEP 6: ROC & PR CALCULATIONS
    # ------------------------------------------------------
    try:
        if not plan:
            echo_message("\nStarting STEP 6: ROC & PR Calculations", verbose)
            roc_output = output
            if use_output_control:
                roc_output = output if output_control.get("write_roc_outputs", False) else None
            roc_results, skipped_info = calculate_roc_metrics(
                df_experiment=exp_full_df,
                df_predictions=pred_full_df,
                cell_line_list=cell_lines,
                threshold=threshold,
                threshold_offsets=threshold_offsets,
                n_bootstrap=roc_bootstrap_n,
                ci_level=roc_bootstrap_ci,
                verbose=verbose,
                output_path=roc_output
            )
            artifacts["roc_results"] = roc_results
            echo_message("\nROC & PR calculations completed successfully.", verbose)
            if skipped_info:
                echo_message(f"Skipped cell lines during ROC & PR calculations: {skipped_info}", verbose)
    except KeyError as e:
        raise RuntimeError(f"Missing key in ROC & PR calculations: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error in STEP 6: {e}") from e
    if stop_after == "roc_metrics":
        echo_message("\nStopping pipeline after STEP 6: ROC & PR Calculations", verbose)
        return artifacts
    if stop_after is None:
        echo_message("\nPipeline completed successfully.", verbose)
        return artifacts

    # ------------------------------------------------------