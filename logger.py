import os
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name="chatbot_logger", log_dir="logs", log_file_name="chatbot.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_file_name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7
        )
        handler.suffix = "%Y-%m-%d.log"
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()
