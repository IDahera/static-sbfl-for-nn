"""
Shared utilities for pipeline scripts.

Provides consistent logging, directory discovery, timing, and iteration helpers
to reduce boilerplate across all scripts.
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

# ============================================================================
# Logging Configuration
# ============================================================================

_SEPARATOR_WIDTH = 60


def get_logger(name: str = None) -> logging.Logger:
    """Get a configured logger with consistent formatting.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def log_section(logger: logging.Logger, title: str, char: str = "=") -> None:
    """Log a major section header.

    Example output:
        ════════════════════════════════════════════════════════════
        Processing config: config1
        ════════════════════════════════════════════════════════════
    """
    separator = char * _SEPARATOR_WIDTH
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_subsection(
    logger: logging.Logger, title: str, char: str = "-"
) -> None:
    """Log a subsection header.

    Example output:
        ────────────────────────────────────────
        Processing dataset: mnist
        ────────────────────────────────────────
    """
    separator = char * 40
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form.

    Automatically chooses the most appropriate unit:
    - Under 1 minute: "45.2s"
    - Under 1 hour: "5m 32s"
    - 1 hour or more: "2h 15m 30s"

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        secs = remaining % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def log_duration(logger: logging.Logger, label: str, seconds: float) -> None:
    """Log a duration with automatic formatting.

    Args:
        logger: Logger instance.
        label: Description of what was timed.
        seconds: Duration in seconds.
    """
    logger.info(f"{label} in {format_duration(seconds)}")


def log_completion(logger: logging.Logger, task: str) -> None:
    """Log a completion message with visual emphasis.

    Example output:
        ════════════════════════════════════════════════════════════
        ✓ Model generation completed for all configs and datasets
        ════════════════════════════════════════════════════════════
    """
    separator = "=" * _SEPARATOR_WIDTH
    logger.info("")
    logger.info(separator)
    logger.info(f"✓ {task}")
    logger.info(separator)


# ============================================================================
# Directory Discovery
# ============================================================================


def find_subdirs(base_path: str, required: bool = True) -> list[str]:
    """Find all subdirectories in a given path.

    Args:
        base_path: Path to search for subdirectories.
        required: If True, logs error when base_path doesn't exist.

    Returns:
        Sorted list of subdirectory names (not full paths).
    """
    logger = get_logger(__name__)

    if not os.path.exists(base_path):
        if required:
            logger.error(f"Directory does not exist: {base_path}")
        return []

    subdirs = [
        d
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    return sorted(subdirs)


def select_subdirs(
    base_path: str,
    selected: Sequence[str] = (),
    required: bool = True,
) -> list[str]:
    """Return all subdirectories, or only explicitly selected ones."""
    available = find_subdirs(base_path, required=required)
    if not selected:
        return available

    selected_set = set(selected)
    return [name for name in available if name in selected_set]


@dataclass
class ConfigDatasetPath:
    """Container for config/dataset iteration context."""

    config_name: str
    dataset_name: str
    config_path: str
    dataset_path: str


def iter_config_datasets(
    base_dir: str, logger: logging.Logger = None
) -> Iterator[ConfigDatasetPath]:
    """Iterate over all config/dataset combinations in a directory structure.

    Expects structure like:
        base_dir/
            config1/
                mnist/
                fmnist/
            config2/
                mnist/
                fmnist/

    Args:
        base_dir: Base directory containing config subdirectories.
        logger: Optional logger for progress messages.

    Yields:
        ConfigDatasetPath objects for each config/dataset combination.
    """
    if logger is None:
        logger = get_logger(__name__)

    config_dirs = find_subdirs(base_dir)

    if not config_dirs:
        logger.warning(f"No config subdirectories found in {base_dir}")
        return

    logger.info(f"Found {len(config_dirs)} config(s): {config_dirs}")

    for config_name in config_dirs:
        config_path = os.path.join(base_dir, config_name)
        log_section(logger, f"Processing config: {config_name}")

        dataset_dirs = find_subdirs(config_path)

        if not dataset_dirs:
            logger.warning(f"No dataset subdirectories in {config_path}")
            continue

        for dataset_name in dataset_dirs:
            dataset_path = os.path.join(config_path, dataset_name)
            log_subsection(logger, f"Processing dataset: {dataset_name}")

            yield ConfigDatasetPath(
                config_name=config_name,
                dataset_name=dataset_name,
                config_path=config_path,
                dataset_path=dataset_path,
            )


# ============================================================================
# Timing Utilities
# ============================================================================


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # do work
        print(f"Took {t.duration:.2f}s")

    Or with auto-logging:
        with Timer(logger, "Model training"):
            # do work
        # Automatically logs: "Model training in 2m 34s"
    """

    def __init__(self, logger: logging.Logger = None, label: str = None):
        """Initialize timer.

        Args:
            logger: If provided, logs duration on exit.
            label: Description for the logged message.
        """
        self.logger = logger
        self.label = label
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        if self.logger and self.label:
            log_duration(self.logger, self.label, self.duration)

    @property
    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer."""
        return time.time() - self.start_time


@contextmanager
def timed_section(
    logger: logging.Logger, section_name: str, is_subsection: bool = False
) -> Iterator[Timer]:
    """Context manager that logs section header and duration.

    Args:
        logger: Logger instance.
        section_name: Name to display in header and completion message.
        is_subsection: If True, uses subsection styling.

    Yields:
        Timer instance for access to elapsed time.

    Example:
        with timed_section(logger, "Processing config: config1"):
            # do work
        # Logs section header, then "Completed in 2m 34s" on exit
    """
    if is_subsection:
        log_subsection(logger, section_name)
    else:
        log_section(logger, section_name)

    timer = Timer()
    timer.start_time = time.time()

    try:
        yield timer
    finally:
        timer.end_time = time.time()
        timer.duration = timer.end_time - timer.start_time
        log_duration(logger, "Completed", timer.duration)
