import subprocess

import pytest
from loguru import logger

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: list[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(reason=msg)


def run_command(
    script: str, args: list[str], hydra_kwargs: dict[str, str], test_name: str, expected_returncode: int = 0
):
    command_parts = [script] + args + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command = " ".join(command_parts)
    logger.info(command)
    command_out = subprocess.run(command, shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != expected_returncode:
        raise AssertionError(
            f"{test_name} returned {command_out.returncode} (expected {expected_returncode})!\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stderr, stdout
