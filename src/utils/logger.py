"""
Centralized logging configuration.

Provides a standard logging setup with both console and rotating file output
for consistent logging across all scripts.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name (typically __name__ or script name)
        log_dir: Directory for log files. If None, only console logging is used.
        level: Logging level (default: INFO)
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger("build_dataset", Path("logs"))
        >>> logger.info("Starting dataset build...")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dated log file: name_YYYY-MM-DD.log
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{name}_{date_str}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger
