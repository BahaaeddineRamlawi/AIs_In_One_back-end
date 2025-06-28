import os
import json
from logger import logger

class PersistentKeyRotator:
    def __init__(self, key1: str, key2: str, index_file: str = "key_index.json"):
        self.keys = [key1, key2]
        self.index_file = index_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f:
                json.dump({"index": 0}, f)
            logger.info(f"[KeyRotator] Created new index file: {self.index_file}")

    def get_current_index(self) -> int:
        try:
            with open(self.index_file, "r") as f:
                index = json.load(f).get("index", 0)
                logger.debug(f"[KeyRotator] Current index: {index}")
                return index
        except Exception as e:
            logger.error(f"[KeyRotator] Failed to read index: {e}")
            return 0

    def get_next_key_index(self) -> int:
        index = self.get_current_index()
        new_index = 1 - index
        try:
            with open(self.index_file, "w") as f:
                json.dump({"index": new_index}, f)
            logger.info(f"[KeyRotator] Switched key index from {index} to {new_index}")
        except Exception as e:
            logger.error(f"[KeyRotator] Failed to write new index: {e}")
        return index
