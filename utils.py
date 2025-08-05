from pathlib import Path
import shutil
import pandas
from typing import Union, Optional, Iterable, List
import fnmatch

PathLike = Union[str, Path]
#///////////////////////////////////////////////////////////////////////////////////////////////////////
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

#///////////////////////////////////////////////////////////////////////////////////////////////////////
def copy_files(
        sources: Iterable[PathLike],
        destination_dir: PathLike,
        file_patterns: Optional[Union[str, List[str]]] = None,
        overwrite: bool = False,
        verbose: bool = False
        ) -> int:
    """
    Copy files from source paths to a destination directory, optionally filtering by patterns.

    Args:
        sources (Iterable[PathLike]): Source paths to copy files from.
        destination (PathLike): Destination directory to copy files to.
        patterns (Optional[Union[str, List[str]]]): Patterns to filter files. If None, all files are copied.
        overwrite (bool): If True, overwrite existing files in the destination.
        verbose (bool): If True, print detailed information about the copying process.
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

            target = dest / path.name
            if target.exists():
                if overwrite:
                    target.unlink()
                else:
                    continue

            shutil.copy2(path, target)
            copied_files += 1
            if verbose:
                print(f"Copied {path} to {target}")
    if verbose:
        print(f"Total files copied: {copied_files}")
    
    return copied_files

#//////////////////////////////////////////////////////////////////////////////////////////////////////////
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