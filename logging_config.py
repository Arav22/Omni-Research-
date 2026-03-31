"""
Production-grade logging configuration for the OmniResearch engine.

Features:
    - Configurable log levels via LOG_LEVEL environment variable
    - Console handler with human-readable colored output
    - Rotating file handler with JSON-structured logs (max 10MB, 5 backups)
    - Per-run correlation IDs for tracing all events in a single research run
    - Module-scoped loggers with hierarchical naming (research.agents, research.search, etc.)
    - Exception chain preservation — full tracebacks stored in file logs

Usage:
    from logging_config import setup_logging, get_logger, set_correlation_id

    setup_logging()                       # Call once at startup
    logger = get_logger(__name__)         # Per-module logger
    set_correlation_id("run-abc123")      # Tag all logs for this run

Environment Variables:
    LOG_LEVEL:   DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO)
    LOG_DIR:     Directory for log files (default: ./logs)
    LOG_JSON:    "true" to enable JSON-formatted console output (default: false)
"""

import os
import sys
import json
import uuid
import logging
import logging.handlers
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Correlation ID — per-run tracing
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="no-correlation-id")


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set a correlation ID for the current execution context.

    Args:
        cid: Custom correlation ID. If None, generates a UUID.

    Returns:
        The correlation ID that was set.
    """
    if cid is None:
        cid = f"run-{uuid.uuid4().hex[:12]}"
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


# ---------------------------------------------------------------------------
# Custom Formatters
# ---------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    """
    Human-readable colored console formatter.

    Maps log levels to ANSI colors for quick visual scanning in terminals.
    Includes timestamp, level, logger name, correlation ID, and message.
    """

    COLORS = {
        logging.DEBUG:    "\033[36m",      # Cyan
        logging.INFO:     "\033[32m",      # Green
        logging.WARNING:  "\033[33m",      # Yellow
        logging.ERROR:    "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m",    # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        cid = get_correlation_id()

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        base = (
            f"{color}{timestamp} "
            f"[{record.levelname:<8}] "
            f"[{cid}] "
            f"{record.name}: "
            f"{record.getMessage()}{self.RESET}"
        )

        if record.exc_info and record.exc_info[1]:
            base += f"\n{color}{self.formatException(record.exc_info)}{self.RESET}"

        return base


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for file logging and log aggregation services.

    Each log line is a self-contained JSON object with:
    - timestamp (ISO 8601), level, logger, correlation_id, message
    - exception details with full traceback when present
    - extra fields from the log record
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": get_correlation_id(),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Preserve full exception chain
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Include any extra fields attached to the record
        for key in ("agent_name", "query", "duration_ms", "search_query",
                     "result_count", "retry_attempt", "status"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Get a namespaced logger under the 'research' hierarchy.

    Args:
        name: Module name (typically __name__). Will be prefixed with 'research.'
              unless it already starts with 'research'.

    Returns:
        Configured logging.Logger instance.
    """
    if not name.startswith("research"):
        name = f"research.{name}"
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_logging_initialized = False


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    json_console: Optional[bool] = None,
) -> None:
    """
    Initialize the logging system. Safe to call multiple times — subsequent
    calls are no-ops.

    Args:
        level: Log level string. Falls back to LOG_LEVEL env var, then INFO.
        log_dir: Log file directory. Falls back to LOG_DIR env var, then ./logs.
        json_console: If True, console output uses JSON format.
                      Falls back to LOG_JSON env var, then False.
    """
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    # Resolve configuration
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_dir = log_dir or os.getenv("LOG_DIR", "logs")
    if json_console is None:
        json_console = os.getenv("LOG_JSON", "false").lower() == "true"

    numeric_level = getattr(logging, level, logging.INFO)

    # Root 'research' logger
    root_logger = logging.getLogger("research")
    root_logger.setLevel(numeric_level)
    root_logger.propagate = False

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    if json_console:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # --- Rotating File Handler ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "research.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # File always captures everything
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # --- Error-only File Handler ---
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "research.error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # Log startup
    startup_logger = get_logger("config")
    startup_logger.info(
        "Logging initialized",
        extra={"status": "startup", "level": level, "log_dir": str(log_path)},
    )


def reset_logging() -> None:
    """Reset logging state (for testing)."""
    global _logging_initialized
    _logging_initialized = False
    root_logger = logging.getLogger("research")
    root_logger.handlers.clear()
