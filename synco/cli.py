import argparse
import json
import sys
from pathlib import Path

try:
    import yaml  # optional dependency
except Exception:
    yaml = None

from .main import run_pipeline


def _load_config(path: Path):
    text = path.read_text(encoding="utf-8")
    lower = path.suffix.lower()
    if lower in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configuration files. Install with 'pip install pyyaml' or use JSON config.")
        return yaml.safe_load(text)
    else:
        # default to JSON
        return json.loads(text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="synco", description="Run SYNCO pipeline from a configuration file or direct CLI args")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", help="Path to pipeline configuration file (JSON or YAML)")
    # direct-args mode: allow specifying minimal required values
    group.add_argument("--base", help="Base path for pipeline data (enables direct-args mode)")

    # Direct args (used when --base is provided)
    p.add_argument("--pipeline-runs", dest="pipeline_runs", help="Path to pipeline runs (when using direct args)")
    p.add_argument("--input", dest="input_path", help="Input folder containing synergies and profiles (when using direct args)")
    p.add_argument("--output", dest="output_path", help="Output folder for results (when using direct args)")
    p.add_argument("--cell-lines", dest="cell_lines", help="Comma-separated list of cell lines to analyze (when using direct args)")
    p.add_argument("--prediction-method", dest="prediction_method", default="DrugLogics", help="Prediction method to use (default: DrugLogics)")

    p.add_argument("--plan", action="store_true", help="Print pipeline plan without executing")
    p.add_argument("--synergies_filename", dest="synergies_filename", help="Exact filename or path to experimental synergies CSV to use (overrides config)")
    p.add_argument("--stop-after", dest="stop_after", choices=[
        "fetch",
        "drug_profiles",
        "synergy_predictions",
        "synergy_convergence",
        "synergy_comparison",
        "roc_metrics",
    ], help="Stop after the specified pipeline step")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Build config either from file or from CLI direct args
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.exists():
            print(f"Config file not found: {cfg_path}", file=sys.stderr)
            return 2

        try:
            config = _load_config(cfg_path)
        except Exception as e:
            print(f"Failed to load configuration: {e}", file=sys.stderr)
            return 3
        # Allow overriding synergies filename from CLI
        if args.synergies_filename:
            config.setdefault('advance', {})
            config['advance'].setdefault('data_loading', {})
            config['advance']['data_loading']['synergy_filename'] = args.synergies_filename
    else:
        # direct-args mode: require minimal fields
        base = Path(args.base).as_posix()
        pipeline_runs = args.pipeline_runs or (str(Path(base) + "/runs"))
        input_path = args.input_path or (str(Path(base) + "/input"))
        output_path = args.output_path or None
        cell_lines = []
        if args.cell_lines:
            cell_lines = [c.strip() for c in args.cell_lines.split(',') if c.strip()]

        config = {
            "paths": {
                "base": base,
                "pipeline_runs": pipeline_runs,
                "input": input_path,
                "output": output_path,
            },
            "general": {
                "cell_lines": cell_lines,
            },
            "compare": {
                "prediction_method": args.prediction_method,
            },
            "advance": {
                "data_loading": {
                    "synergy_filename": args.synergies_filename if args.synergies_filename else None
                }
            }
        }

    try:
        run_pipeline(config=config, plan=args.plan, stop_after=args.stop_after, verbose=args.verbose)
        return 0
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
