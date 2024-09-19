import os
import re
from importlib.resources import files

import hydra
from omegaconf import DictConfig

config_yaml = files("meds_torch").joinpath("configs/latest_dir.yaml")


def get_latest_directory(path):
    """Find the latest directory in the given path based on timestamp in the directory name.

    The function expects directory names in the format 'YYYY-MM-DD_HH-MM-SS-microseconds'.
    It returns the full path of the directory with the latest timestamp.

    Args:
    path (str): The path to search for directories.

    Returns:
        str or None: Full path of the latest directory, or None if no directories are found.
    Raises:
        ValueError: If no directories are found in the given path.
        ValueError: If any directory name doesn't match the expected timestamp format.

    Examples:
    >>> import tempfile
    >>> import pytest
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create test directories
    ...     os.mkdir(os.path.join(tmpdir, '2024-09-14_15-18-59_216694'))
    ...     os.mkdir(os.path.join(tmpdir, '2024-09-14_16-20-00_000000'))
    ...     os.mkdir(os.path.join(tmpdir, '2024-09-15_10-30-00_123456'))
    ...     latest = get_latest_directory(tmpdir)
    ...     os.path.basename(latest)
    '2024-09-15_10-30-00_123456'
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Test with no directories
    ...     with pytest.raises(ValueError):
    ...         get_latest_directory(tmpdir)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Test with non-standard named directory
    ...     os.mkdir(os.path.join(tmpdir, 'not_a_timestamp'))
    ...     os.mkdir(os.path.join(tmpdir, '2024-09-14_15-18-59_216694'))
    ...     with pytest.raises(ValueError):
    ...         get_latest_directory(tmpdir)
    """

    def to_int(dir_name):
        # Remove '-' and '_', then convert to integer
        return int("".join(c for c in dir_name if c.isdigit()))

    def is_valid_timestamp(dir_name):
        pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+$"
        return bool(re.match(pattern, dir_name))

    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not directories:
        raise ValueError(f"No directories found in {path}")

    invalid_dirs = [d for d in directories if not is_valid_timestamp(d)]
    if invalid_dirs:
        raise ValueError(f"Invalid directory names found: {', '.join(invalid_dirs)}")

    latest = max(directories, key=to_int)
    return os.path.join(path, latest)


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    print(get_latest_directory(cfg.path))


if __name__ == "__main__":
    main()
