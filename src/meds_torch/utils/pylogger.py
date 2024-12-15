from collections.abc import Callable
from functools import wraps
from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger


class RankedLogger:
    """A multi-GPU-friendly logger based on loguru that adds rank information to messages.

    Examples:
    >>> caplog = getfixture('caplog')
    >>> from lightning_utilities.core.rank_zero import rank_zero_only
    >>>
    >>> # Set up the rank for testing
    >>> rank_zero_only.rank = 0
    >>>
    >>> # Create logger instance
    >>> log = RankedLogger(rank_zero_only=False)
    >>>
    >>> # Test logging
    >>> log.info("This is an info message")
    >>> assert "[rank0] This is an info message" in caplog.text
    >>>
    >>> log.debug("This is a debug message", rank=0)
    >>> assert "[rank0] This is a debug message" in caplog.text
    >>>
    >>> # Test rank zero only mode
    >>> log_zero = RankedLogger(rank_zero_only=True)
    >>> log_zero.info("This only logs on rank 0")
    >>> assert "[rank0] This only logs on rank 0" in caplog.text
    """

    def __init__(
        self,
        name: str = "ranked_logger",
        rank_zero_only: bool = False,
    ) -> None:
        """Initialize the ranked logger.

        Args:
            name: Name identifier for the logger
            rank_zero_only: Whether to force all logs to only occur on rank zero process
        """
        self.name = name
        self.rank_zero_only = rank_zero_only

    def _format_message(self, message: str) -> str:
        """Add rank prefix to message."""
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        return f"[rank{current_rank}] {message}"

    def _log_decorator(func: Callable) -> Callable:
        """Decorator to handle rank-specific logging."""

        @wraps(func)
        def wrapper(self, *args, rank: int | None = None, **kwargs) -> Any:
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")

            if self.rank_zero_only and current_rank != 0:
                return

            if rank is not None and current_rank != rank:
                return

            # Format the message with rank information
            if args:
                args = list(args)
                args[0] = self._format_message(args[0])

            return func(self, *args, **kwargs)

        return wrapper

    @_log_decorator
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        logger.debug(message, *args, **kwargs)

    @_log_decorator
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        logger.info(message, *args, **kwargs)

    @_log_decorator
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        logger.warning(message, *args, **kwargs)

    @_log_decorator
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        logger.error(message, *args, **kwargs)

    @_log_decorator
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        logger.critical(message, *args, **kwargs)
