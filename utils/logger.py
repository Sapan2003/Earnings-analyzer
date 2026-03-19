import logging
import os


def get_logger(name: str, log_file: str) -> logging.Logger:
    """
    Creates and returns a logger that writes to both
    a log file and the terminal simultaneously.

    Args:
        name: Name of the logger (usually the module name)
        log_file: Name of the log file (e.g. 'ingestion.log')

    Returns:
        Configured logger instance
    """

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # --- File Handler ---
    # Saves ALL log levels (DEBUG and above) to file
    file_handler = logging.FileHandler(f"logs/{log_file}")
    file_handler.setLevel(logging.DEBUG)

    # --- Console Handler ---
    # Shows only INFO and above in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # --- Format ---
    # Example: 2026-03-19 10:23:41 | INFO | sec_fetcher | message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger