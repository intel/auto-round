import logging
import os
from io import StringIO

from auto_round.logger import TRACE_LEVEL, AutoRoundFormatter, logger


def test_logger(monkeypatch):
    # Mock the AR_LOG_LEVEL environment variable
    monkeypatch.setenv("AR_LOG_LEVEL", "TRACE")

    # Create a StringIO to capture log output
    log_output = StringIO()
    stream_handler = logging.StreamHandler(log_output)
    stream_handler.setFormatter(AutoRoundFormatter())

    # Add the handler to the logger
    logger.addHandler(stream_handler)
    logger.setLevel(logging.getLevelName(os.getenv("AR_LOG_LEVEL", "INFO")))

    # Log messages at different levels
    logger.trace("This is a TRACE message.")
    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")

    # Test warning_once functionality
    logger.warning_once("This is a WARNING_ONCE message.")
    logger.warning_once("This is a WARNING_ONCE message.")  # Should not log again
    logger.warning_once("This is another unique WARNING_ONCE message.")  # Should log

    # Remove the handler after the test
    logger.removeHandler(stream_handler)

    # Get the log output
    log_output.seek(0)
    logs = log_output.read()

    # Assertions for log levels
    assert "TRACE" in logs
    assert "This is a TRACE message." in logs
    assert "DEBUG" in logs
    assert "This is a DEBUG message." in logs
    assert "INFO" in logs
    assert "This is an INFO message." in logs
    assert "WARNING" in logs
    assert "This is a WARNING message." in logs
    assert "ERROR" in logs
    assert "This is an ERROR message." in logs
    assert "CRITICAL" in logs
    assert "This is a CRITICAL message." in logs

    # Assertions for warning_once
    assert logs.count("This is a WARNING_ONCE message.") == 1
    assert "This is another unique WARNING_ONCE message." in logs
