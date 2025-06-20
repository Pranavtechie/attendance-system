from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# =============================================================================
# Centralised logging configuration for the entire application
# =============================================================================
from src.config import DEFAULT_LOG_FILE, LOG_DIR


def setup_logging(
    *,
    level: int = logging.INFO,
    console_level: int | None = logging.WARNING,
    log_file: Path | str = DEFAULT_LOG_FILE,
    max_mb: int = 10,
    backups: int = 5,
) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    level : int, optional
        Log level for the file handler (defaults to logging.INFO).
    console_level : int | None, optional
        If not ``None``, add a console (``StreamHandler``) at this level. Use
        ``None`` to completely silence stdout/stderr logging. Defaults to
        ``logging.WARNING`` (so only warnings+ appear in the terminal).
    log_file : Path | str, optional
        Path to the log file (rotating).
    max_mb : int, optional
        Maximum size in megabytes for a single log file before rotation.
    backups : int, optional
        Number of rotated backup files to keep.
    """

    log_file = Path(log_file)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Handler: Rotating file
    # ---------------------------------------------------------------------
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backups,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    handlers: list[logging.Handler] = [file_handler]

    # ---------------------------------------------------------------------
    # Handler: Console (optional)
    # ---------------------------------------------------------------------
    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level)
        handlers.append(console_handler)

    # ---------------------------------------------------------------------
    # Apply configuration. ``force=True`` overrides any prior config that
    # might have been set by imported libraries.
    # ---------------------------------------------------------------------
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Make root logger a bit more chatty about the setup done once.
    logging.getLogger(__name__).debug(
        "Logging initialised → file='%s' | level=%s | console_level=%s",
        log_file,
        logging.getLevelName(level),
        "OFF" if console_level is None else logging.getLevelName(console_level),
    )
