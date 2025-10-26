"""
Comprehensive logging configuration for MCP Obsidian Server.
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation

        return json.dumps(log_data)


class AuditLogger:
    """Audit logger for tracking file operations."""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            log_file: Optional path to audit log file
        """
        self.logger = logging.getLogger("obsidian.audit")
        self.logger.setLevel(logging.INFO)

        # Create audit log handler
        if log_file:
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def log_operation(
        self,
        operation: str,
        path: str,
        user_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a file operation.

        Args:
            operation: Type of operation (read, create, update, delete)
            path: Path to the file
            user_id: Optional user identifier
            success: Whether operation was successful
            details: Optional additional details
        """
        log_data = {
            "operation": operation,
            "path": path,
            "success": success,
            "user_id": user_id or "unknown",
        }

        if details:
            log_data.update(details)

        if success:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.warning(json.dumps(log_data))


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "json",
    audit_enabled: bool = True,
) -> AuditLogger:
    """
    Set up comprehensive logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Format type: "json" or "text"
        audit_enabled: Whether to enable audit logging

    Returns:
        AuditLogger instance
    """
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Create audit logger
    audit_log_file = None
    if audit_enabled and log_file:
        audit_log_file = str(Path(log_file).parent / "audit.log")

    audit_logger = AuditLogger(audit_log_file)

    # Log startup
    logging.info(f"Logging initialized: level={log_level}, format={log_format}")

    return audit_logger
