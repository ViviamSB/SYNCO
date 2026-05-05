from pathlib import Path
import shutil
import pandas
import json
import ast
from typing import Union, Optional, Iterable, List, Sequence, Any
import fnmatch
import logging

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# FILE SYSTEM UTILITIES
# FUNCTIONS: ensure_directory() create_cell_line_dir() get_output_file()
#            copy_files() load_dataframe() save_file() echo_message()
#///////////////////////////////////////////////////////////////////////////////////////////////////////

PathLike = Union[str, Path]
def ensure_directory(path: PathLike, reset: bool = False) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (PathLike): The path to the directory to ensure.
    
    Returns:
        The path object of the directory.        
    """
    p = Path(path)
    if reset and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

#/////////////////////////////////////////////////////
def create_cell_line_dir(cell_line: str, output_directory: PathLike) -> Path:
    """
    Create a directory for a specific cell line within the output directory.
    
    Args:
        cell_line (str): The name of the cell line.
        output_directory (PathLike): The base output directory.
    
    Returns:
        The path object of the created cell line directory.
    """
    cell_line_dir = ensure_directory(Path(output_directory) / cell_line)
    return cell_line_dir

#/////////////////////////////////////////////////////
def get_output_file(cell_line_dir: PathLike, file_name: str) -> Path:
    """
    Get the full path for an output file in the cell line directory.
    """
    return Path(cell_line_dir) / file_name

#/////////////////////////////////////////////////////
def copy_files(
        sources: Iterable[PathLike],
        destination_dir: PathLike,
        file_patterns: Optional[Union[str, List[str]]] = None,
        overwrite: bool = False,
        verbose: bool = False,
        run_date_filter: Optional[str] = None
        ) -> int:
    """
    Copy files from source paths to a destination directory, optionally filtering by patterns.

    Args:
        sources (Iterable[PathLike]): Source paths to copy files from.
        destination (PathLike): Destination directory to copy files to.
        patterns (Optional[Union[str, List[str]]]): Patterns to filter files. If None, all files are copied.
        overwrite (bool): If True, overwrite existing files in the destination.
        verbose (bool): If True, print detailed information about the copying process.
        run_date_filter (Optional[str]): If provided, only copy files from subdirectories containing this date string.
    """
    # Normalize patterns to a list
    patterns: List[str]
    if isinstance(file_patterns, str):
        patterns = [file_patterns]
    elif file_patterns is None:
        patterns = ['*']  # Match all files
    else:
        patterns = file_patterns

    dest = Path(destination_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied_files = 0

    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"{src_path} does not exist")

        to_copy = [src_path]
        if src_path.is_dir():
            to_copy = list(src_path.rglob('*'))

        for path in to_copy:
            if not path.is_file():
                continue

            # If patterns specified, skip files that match none
            if patterns and not any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
                continue

            # Apply run_date_filter if specified
            if run_date_filter:
                relative_path = path.relative_to(src_path)
                parent_dirs = relative_path.parts[:-1]  # Get all parent directories
                
                # For files in root directory (like observed_synergies), always copy
                if len(parent_dirs) == 0:
                    pass  # Copy this file
                # For files in subdirectories, only copy if directory contains the run_date
                elif any(run_date_filter in dir_name for dir_name in parent_dirs):
                    pass  # Copy this file
                else:
                    continue  # Skip this file

            target = dest / path.name
            if target.exists():
                if overwrite:
                    target.unlink()
                else:
                    continue

            shutil.copy2(path, target)
            copied_files += 1
            if verbose:
                print(f"Copied {patterns} to {dest}")
    if verbose:
        print(f"Total files copied: {copied_files}")
    
    return copied_files

#/////////////////////////////////////////////////////
def load_dataframe(
        folder: PathLike,
        pattern: str,
        **read_kwargs: Optional[dict]
        ) -> pandas.DataFrame:
    """
    Load dataframes from files in a specified folder that match given patterns.
    Read each matching file into a pandas DataFrame and return a list of DataFrames.
    
    Args:
        folder (PathLike): The folder containing the file to load.
        pattern (str): Pattern to match the file name.
        **read_kwargs: Additional keyword arguments for pandas read functions.
    Returns:
        DataFrame: A DataFrame loaded from the matching file.
    """
    folder_path = Path(folder)
    matches = sorted(folder_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {folder_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files matching pattern '{pattern}' found in {folder_path}")

    df = pandas.read_csv(matches[0], **read_kwargs)
    return df

#/////////////////////////////////////////////////////
def save_file(
        data,
        output_path: str,
        file_type: str = 'csv',
        **kwargs
) -> None:
    """
    Save data to a file in the specified format.

    Args:
        data: The data to save (DataFrame, dictionary, or txt).
        output_path: The path to the output file.
        file_type: The type of file to save (e.g., 'csv', 'json').
        **kwargs: Additional arguments to pass to the save function.
    """
    if file_type == 'csv':
        if isinstance(data, pandas.DataFrame):
            data.to_csv(output_path, **kwargs)
            print(f'Data saved to {output_path} as CSV.')
        else:
            raise ValueError('For CSV, data should be a pandas DataFrame.')
    elif file_type == 'json':
        if isinstance(data, (pandas.DataFrame, dict)):
            with open(output_path, 'w') as file:
                json.dump(data.to_dict(orient='records') if isinstance(data, pandas.DataFrame) else data, file, **kwargs)
                print(f'Data saved to {output_path} as JSON.')
        else:
            raise ValueError('For JSON, data should be a DataFrame or dictionary.')
    elif file_type == 'txt':
        if isinstance(data, (pandas.DataFrame, dict, str)):
            with open(output_path, 'w') as f:
                if isinstance(data, pandas.DataFrame):
                    f.write(data.to_string(**kwargs))
                elif isinstance(data, dict):
                    json.dump(data, f, **kwargs)
                else:
                    f.write(str(data))
        else:
            raise ValueError('For TXT, data should be a DataFrame, dictionary, or string.')
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
#/////////////////////////////////////////////////////
def echo_message(message: str, verbose: bool):
    if verbose:
        print(message)


def save_fig(
        fig,
        output_dir: PathLike,
        basename: str,
    formats: Optional[Iterable[str]] = None,
    scale: int = 3,
    verbose: bool = False,
    fig_type: Optional[str] = None,
    ) -> dict:
    """Save a Plotly figure in multiple formats safely.

    Args:
        fig: A Plotly figure object (plotly.graph_objects.Figure).
        output_dir: Directory to save the files into.
        basename: Base filename (no extension).
        formats: Iterable of formats to save. Supported: 'png','svg','html'.
        scale: Image scale multiplier for raster exports.
        verbose: Print progress when True.

    Returns:
        dict mapping format -> bool (success).
    """
    from pathlib import Path

    logger = logging.getLogger(__name__)
    supported = {'png', 'svg', 'html'}
    if formats is None:
        formats = ['png', 'html']

    out = {}
    outdir = ensure_directory(output_dir)

    from pathlib import Path

    # Determine backend: allow explicit override via `fig_type`.
    if fig_type is not None:
        fig_type = fig_type.lower()
    is_matplotlib = (fig_type == 'matplotlib') or (fig_type is None and hasattr(fig, 'savefig'))
    is_plotly = (fig_type == 'plotly') or (fig_type is None and (hasattr(fig, 'write_image') or hasattr(fig, 'write_html')))

    for fmt in formats:
        fmt = fmt.lower()
        if fmt not in supported:
            out[fmt] = False
            if verbose:
                logger.warning("Unsupported format: %s", fmt)
            continue

        dest = Path(outdir) / f"{basename}.{fmt}"
        try:
            if is_plotly and not is_matplotlib:
                # Plotly figure: use its writers
                if fmt in ('png', 'svg'):
                    fig.write_image(str(dest), scale=scale)
                elif fmt == 'html':
                    fig.write_html(str(dest))
            elif is_matplotlib and not is_plotly:
                # Matplotlib Figure
                if fmt in ('png', 'svg'):
                    dpi = 100 * int(scale)
                    fig.savefig(str(dest), dpi=dpi, bbox_inches='tight')
                elif fmt == 'html':
                    # Create a PNG first and reference it from a simple HTML file
                    png_path = Path(outdir) / f"{basename}.png"
                    dpi = 100 * int(scale)
                    if not png_path.exists():
                        fig.savefig(str(png_path), dpi=dpi, bbox_inches='tight')
                    html_content = f"<html><body><img src=\"{png_path.name}\" alt=\"{basename}\"></body></html>"
                    with open(dest, 'w', encoding='utf8') as fh:
                        fh.write(html_content)
            else:
                # Unknown figure type: try common methods
                if fmt in ('png', 'svg'):
                    # Try matplotlib-style
                    if hasattr(fig, 'savefig'):
                        dpi = 100 * int(scale)
                        fig.savefig(str(dest), dpi=dpi, bbox_inches='tight')
                    elif hasattr(fig, 'write_image'):
                        fig.write_image(str(dest), scale=scale)
                    else:
                        raise RuntimeError('Figure object does not provide a supported save method')
                elif fmt == 'html':
                    if hasattr(fig, 'write_html'):
                        fig.write_html(str(dest))
                    else:
                        # fall back to writing a PNG and referencing it
                        png_path = Path(outdir) / f"{basename}.png"
                        if hasattr(fig, 'savefig'):
                            dpi = 100 * int(scale)
                            fig.savefig(str(png_path), dpi=dpi, bbox_inches='tight')
                        elif hasattr(fig, 'write_image'):
                            fig.write_image(str(png_path), scale=scale)
                        html_content = f"<html><body><img src=\"{png_path.name}\" alt=\"{basename}\"></body></html>"
                        with open(dest, 'w', encoding='utf8') as fh:
                            fh.write(html_content)

            out[fmt] = True
            if verbose:
                logger.info("Saved %s", dest)
        except Exception as exc:
            out[fmt] = False
            if verbose:
                logger.warning("Failed to save %s: %s", dest, exc)

    return out

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# DATAFRAME UTILITIES
# FUNCTIONS: split_column() apply_mapping() deduplicate_list_column() flag_matches()
#            filter_synergies() make_dictionary() deep_merge()
#///////////////////////////////////////////////////////////////////////////////////////////////////////

def split_column(
        df: pandas.DataFrame,
        source_col: str,
        new_cols: List[str],
        separator: str = '-'
        ) -> pandas.DataFrame:
    """
    Split a column in a DataFrame into multiple columns based on a separator.
    """
    df = df.copy()
    # Split the source column into new columns
    df[new_cols] = df[source_col].str.split(separator, expand=True)
    return df

#/////////////////////////////////////////////////////
def apply_mapping(
        df: pandas.DataFrame,
        key_col: str,
        new_col: str,
        dictionary: Optional[dict] = None,
        mapping_df: Optional[pandas.DataFrame] = None,
        mapping_indexcol: Optional[str] = None,
        mapping_valuecol: Optional[str] = None
        ) -> pandas.DataFrame:
    """
    Apply a mapping to a DataFrame column using a dictionary or another DataFrame.
    """
    df = df.copy()
    if dictionary is not None:
        df[new_col] = df[key_col].map(dictionary)
    elif mapping_df is not None:
        if mapping_indexcol is None or mapping_valuecol is None:
            raise ValueError("If using mapping_df, both mapping_indexcol and mapping_valuecol must be specified.")
        mapping_dict = mapping_df.set_index(mapping_indexcol)[mapping_valuecol].to_dict()
        df[new_col] = df[key_col].map(mapping_dict)
    else:
        raise ValueError("Either a dictionary or a mapping DataFrame must be provided.")
    return df

#/////////////////////////////////////////////////////
def deduplicate_list_column(
        df: pandas.DataFrame,
        column: str,
        as_string: bool = False,
        separator: str = ', '
        ) -> pandas.DataFrame:
    """
    Deduplicate entries in a DataFrame column that contains lists or strings of items.
    If value in column is None, it will remain None.
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    df[column] = df[column].apply(lambda x: sorted(set(x), key=x.index) if isinstance(x, list) else x if x is not None else None)
    if as_string:
        df[column] = df[column].apply(lambda lst: separator.join(lst) if isinstance(lst, list) else lst if lst is not None else None)
    return df

#/////////////////////////////////////////////////////
def flag_matches(
        df: pandas.DataFrame,
        key_col: str,
        new_col: str,
        match_values: Sequence,
        true_values: Any = 1,
        false_values: Any = 0
        ) -> pandas.Series:
    """
    Flag matches in a DataFrame column based on a list of values.
    """
    df = df.copy()
    df[new_col] = df[key_col].apply(lambda x: true_values if x in match_values else false_values)
    return df[new_col]

#/////////////////////////////////////////////////////
def filter_synergies(
        cell_line: str,
        df: pandas.DataFrame,
        column_cell_lines: str,
        column_synergies: str
        ) -> pandas.DataFrame:
    """
    Filter a DataFrame to only include rows where the cell line matches & synergies are True (not None)
    and the synergies column is not empty.
    """
    df = df.copy()
    df = df[(df[column_cell_lines] == cell_line) & (df[column_synergies].notna())]
    return df



#/////////////////////////////////////////////////////
def make_dictionary(
        df: pandas.DataFrame,
        key_col: str,
        value_col: str,
        long: Optional[str] = None, # Can be "values" or "keys" to flatten and deduplicate lists
        ) -> dict:
    """
    Create a dictionary from two columns of a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        key_col (str): Column name to use as keys in the dictionary.
        value_col (str): Column name to use as values in the dictionary.
        long (bool): If True, flatten and deduplicate list values in value_col.
    Returns:
        dict: Dictionary with keys and values from the specified columns.
    """
    if key_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns '{key_col}' or '{value_col}' not found in DataFrame.")
    
    if long == "values":
        df[value_col] = df[value_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        df = df.explode(value_col)
    if long == "keys":
        df[key_col] = df[key_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        df = df.explode(key_col)

    # Remove NaN values
    df = df.dropna(subset=[key_col, value_col])

    # Make dictionary
    dictionary = df.set_index(key_col)[value_col].to_dict()

    return dictionary

#/////////////////////////////////////////////////////
def deep_merge(destination, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_merge(destination[key], value)
        else:
            destination[key] = value
    return destination

#/////////////////////////////////////////////////////
def clean_cell_names(
        df: pandas.DataFrame,
        column: str = 'cell_line'
) -> pandas.DataFrame:
    """
    Clean cell names in a DataFrame by removing unwanted characters.
    """
    df = df.copy()
    df[column] = df[column].str.upper().str.replace('-', '', regex=True)
    return df