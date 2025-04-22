# file_utils.py
import os
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


@contextmanager
def try_lock(path: str | Path, timeout: float = 0, overwrite: bool = False):
    if isinstance(path, Path):
        path = str(path.resolve())
    lockfile = path + ".lock"
    lock = FileLock(lockfile, timeout=timeout)

    acquired = False
    try:
        lock.acquire()
        acquired = True
        # Run the code in the context manager if overwrite is true or the output file does not exist
        if os.path.exists(path) and not overwrite:
            yield False
        else:
            yield True
    except Timeout:
        # Don't execute code in the context manager if the lock is not acquired
        yield False
    finally:
        # release lock if acquired and delete the lock files
        if acquired:
            lock.release()
            try:
                os.remove(lockfile)
            except OSError:
                pass
