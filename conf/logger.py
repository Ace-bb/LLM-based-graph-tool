import logging
import os

class LogColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


class ColorfulFormatter(logging.Formatter):
    """Custom formatter to add colors for console logs."""

    def format(self, record):
        log_colors = {
            logging.DEBUG: LogColors.CYAN,
            logging.INFO: LogColors.GREEN,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
            logging.CRITICAL: LogColors.MAGENTA,
        }
        color = log_colors.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
        record.msg = f"{color}{record.msg}{LogColors.RESET}"
        return super().format(record)


def setup_logger(log_file, log_level):
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Shared file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the root logger
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():  # Prevent duplicate handlers
        root_logger.setLevel(getattr(logging, log_level))
        root_logger.addHandler(file_handler)

    # Console handler with colorful formatter for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = ColorfulFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)
