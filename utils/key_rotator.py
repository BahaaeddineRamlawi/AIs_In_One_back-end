import os
import json

class PersistentKeyRotator:
    def __init__(self, key1, key2, index_file="key_index.json"):
        self.keys = [key1, key2]
        self.index_file = index_file
        self.index = self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                try:
                    return json.load(f).get("index", 0) % len(self.keys)
                except:
                    return 0
        return 0

    def save_index(self):
        with open(self.index_file, "w") as f:
            json.dump({"index": self.index}, f)

    def get_current_key(self):
        return self.keys[self.index]

    def rotate_key(self):
        prev = self.index
        self.index = (self.index + 1) % len(self.keys)
        self.save_index()
        print(f"[KeyRotator] Switched key index from {prev} to {self.index}")
