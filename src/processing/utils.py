import logging
import multiprocessing

MONTHS_MAP = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12
    }

def setup_logger(log_file, log_level=logging.INFO, write_console=True, write_file=True) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    if write_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if write_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def get_optimal_process_count() -> int:
    """
    Calculate optimal process count for CPU-bound PDF processing.
    ProcessPoolExecutor uses true parallelism with multiprocessing (avoids GIL).
    """
    cpu_count = multiprocessing.cpu_count()
    optimal_procs = max(2, cpu_count)
    return optimal_procs