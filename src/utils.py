import pathlib
from typing import List

def create_experiment_log_dir(root: str, parents: bool = True) -> str:
    """
    Create a directory for logging experiments.

    Parameters:
    root (str): The root directory where experiment logs will be stored.
    parents (bool): Whether to create parent directories if they do not exist.

    Returns:
    str: The path to the created experiment log directory.
    """
    root_path = pathlib.Path(root).resolve()
    child = (
        create_from_missing(root_path)
        if not root_path.exists()
        else create_from_existing(root_path)
    )
    child.mkdir(parents=parents)
    return child.as_posix()

def create_from_missing(root: pathlib.Path) -> pathlib.Path:
    """
    Create the initial subdirectory if the root directory does not exist.

    Parameters:
    root (pathlib.Path): The root directory path.

    Returns:
    pathlib.Path: The path to the created subdirectory.
    """
    return root / "0"

def create_from_existing(root: pathlib.Path) -> pathlib.Path:
    """
    Create a new subdirectory based on existing subdirectories.

    Parameters:
    root (pathlib.Path): The root directory path.

    Returns:
    pathlib.Path: The path to the created subdirectory.
    """
    children = [
        int(c.name) for c in root.glob("*")
        if (c.is_dir() and c.name.isnumeric())
    ]
    if is_first_experiment(children):
        child = create_from_missing(root)
    else:
        child = root / increment_experiment_number(children)
    return child

def is_first_experiment(children: List[int]) -> bool:
    """
    Check if there are no existing subdirectories.

    Parameters:
    children (List[int]): List of existing subdirectory numbers.

    Returns:
    bool: True if there are no existing subdirectories, False otherwise.
    """
    return len(children) == 0

def increment_experiment_number(children: List[int]) -> str:
    """
    Increment the experiment number based on existing subdirectories.

    Parameters:
    children (List[int]): List of existing subdirectory numbers.

    Returns:
    str: The incremented experiment number as a string.
    """
    return str(max(children) + 1)
