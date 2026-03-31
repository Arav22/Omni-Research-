"""
Tests for the logging configuration module.
"""

import json
import logging
import pytest

from logging_config import (
    setup_logging,
    reset_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    ConsoleFormatter,
    JSONFormatter,
)


class TestCorrelationId:
    """Tests for per-run correlation ID management."""

    def test_default_correlation_id(self):
        """Should have a default value when not explicitly set."""
        cid = get_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_set_custom_correlation_id(self):
        """Setting a custom ID should return and store it."""
        cid = set_correlation_id("test-run-123")
        assert cid == "test-run-123"
        assert get_correlation_id() == "test-run-123"

    def test_auto_generated_correlation_id(self):
        """Passing None should generate a UUID-based ID."""
        cid = set_correlation_id(None)
        assert cid.startswith("run-")
        assert len(cid) == 16  # "run-" + 12 hex chars


class TestGetLogger:
    """Tests for the logger factory function."""

    def test_logger_name_prefixed(self):
        """Logger name should be prefixed with 'research.' namespace."""
        logger = get_logger("search")
        assert logger.name == "research.search"

    def test_logger_already_prefixed(self):
        """Should not double-prefix if name already starts with 'research'."""
        logger = get_logger("research.workflow")
        assert logger.name == "research.workflow"

    def test_returns_logging_logger(self):
        """Should return a standard logging.Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)


class TestJSONFormatter:
    """Tests for the structured JSON log formatter."""

    def test_basic_format(self):
        """Should produce valid JSON with required fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="research.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "research.test"
        assert parsed["message"] == "Test message"
        assert parsed["line"] == 42
        assert "timestamp" in parsed
        assert "correlation_id" in parsed

    def test_exception_info_included(self):
        """Should include exception details when exc_info is present."""
        formatter = JSONFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="research.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=50,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert "test error" in parsed["exception"]["message"]
        assert isinstance(parsed["exception"]["traceback"], list)

    def test_extra_fields_included(self):
        """Should include recognized extra fields in the output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="research.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Search done",
            args=(),
            exc_info=None,
        )
        record.agent_name = "Claude"
        record.duration_ms = 1500

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["agent_name"] == "Claude"
        assert parsed["duration_ms"] == 1500


class TestSetupLogging:
    """Tests for the logging initialization."""

    def setup_method(self):
        reset_logging()

    def teardown_method(self):
        reset_logging()

    def test_setup_creates_handlers(self, tmp_path):
        """setup_logging should add handlers to the research root logger."""
        setup_logging(level="DEBUG", log_dir=str(tmp_path))

        root_logger = logging.getLogger("research")
        assert len(root_logger.handlers) >= 2  # Console + at least one file

    def test_setup_is_idempotent(self, tmp_path):
        """Calling setup_logging twice should not duplicate handlers."""
        setup_logging(level="INFO", log_dir=str(tmp_path))
        handler_count = len(logging.getLogger("research").handlers)

        setup_logging(level="DEBUG", log_dir=str(tmp_path))
        assert len(logging.getLogger("research").handlers) == handler_count

    def test_log_files_created(self, tmp_path):
        """Should create log files in the specified directory."""
        setup_logging(level="DEBUG", log_dir=str(tmp_path))

        logger = get_logger("test")
        logger.info("Test log entry")

        # Force flush
        for handler in logging.getLogger("research").handlers:
            handler.flush()

        log_files = list(tmp_path.iterdir())
        log_names = [f.name for f in log_files]
        assert "research.log" in log_names
        assert "research.error.log" in log_names

    def test_reset_clears_handlers(self, tmp_path):
        """reset_logging should remove all handlers."""
        setup_logging(level="INFO", log_dir=str(tmp_path))
        reset_logging()

        root_logger = logging.getLogger("research")
        assert len(root_logger.handlers) == 0
