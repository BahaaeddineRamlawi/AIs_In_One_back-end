import requests
import os
from dotenv import load_dotenv

from utils.key_rotator import PersistentKeyRotator

load_dotenv()

INDEX_FILE = "key_index.json"

gemini_rotator = PersistentKeyRotator(
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    index_file=INDEX_FILE,
)

def generate_image_base64(prompt: str) -> str:
    current_index = gemini_rotator.index
    gemini_key = gemini_rotator.keys[current_index]

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_key,
    }

    json_data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    }

    response = requests.post(url, headers=headers, json=json_data)
    response.raise_for_status()
    data = response.json()

    image_b64 = None
    for part in data["candidates"][0]["content"]["parts"]:
        if "inlineData" in part and "data" in part["inlineData"]:
            image_b64 = part["inlineData"]["data"]
            break

    if image_b64 is None:
        raise ValueError("No image data found in API response")

    gemini_rotator.rotate_key()

    return image_b64
