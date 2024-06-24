from pathlib import Path


def list_subdir_files(root: Path | str, ext: str) -> list[Path]:
    """List files in subdirectories of a directory with a given extension.

    Args:
        root: Path to the directory.
        ext: File extension to filter files.

    Returns:
        An alphabetically sorted list of Path objects to files matching the extension in any level of
        subdirectories of the given directory.

    Examples:
        >>> import tempfile
        >>> tmpdir = tempfile.TemporaryDirectory()
        >>> root = Path(tmpdir.name)
        >>> subdir_1 = root / "subdir_1"
        >>> subdir_1.mkdir()
        >>> subdir_2 = root / "subdir_2"
        >>> subdir_2.mkdir()
        >>> subdir_1_A = subdir_1 / "A"
        >>> subdir_1_A.mkdir()
        >>> (root / "1.csv").touch()
        >>> (root / "foo.parquet").touch()
        >>> (root / "2.csv").touch()
        >>> (root / "subdir_1" / "3.csv").touch()
        >>> (root / "subdir_2" / "4.csv").touch()
        >>> (root / "subdir_1" / "A" / "5.csv").touch()
        >>> (root / "subdir_1" / "A" / "15.csv.gz").touch()
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "csv")] # doctest: +NORMALIZE_WHITESPACE
        [PosixPath('1.csv'),
         PosixPath('2.csv'),
         PosixPath('subdir_1/3.csv'),
         PosixPath('subdir_1/A/5.csv'),
         PosixPath('subdir_2/4.csv')]
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "parquet")]
        [PosixPath('foo.parquet')]
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "csv.gz")]
        [PosixPath('subdir_1/A/15.csv.gz')]
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "json")]
        []
        >>> list_subdir_files(root / "nonexistent", "csv")
        []
        >>> tmpdir.cleanup()
    """

    return sorted(list(Path(root).glob(f"**/*.{ext}")))