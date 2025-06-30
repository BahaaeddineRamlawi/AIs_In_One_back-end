import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name="chatbot_logger", log_dir="logs", log_file_name="chatbot.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_file_name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()
