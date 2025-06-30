import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from g4f.client import Client
from utils.key_rotator import PersistentKeyRotator

load_dotenv()

INDEX_FILE = "key_index.json"
gemini_rotator = PersistentKeyRotator(
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    index_file=INDEX_FILE,
)

genai.configure(api_key=gemini_rotator.keys[gemini_rotator.index])

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
            {"parts": [{"text": prompt}]}
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


def process_query_image(model, prompt):
    try:
        client = Client()
        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format="url"
        )
        image_url = response.data[0].url
        print(f"[{model}] Generated image URL: {image_url}")
        return image_url
    except Exception as e:
        print(f"[{model}] Error generating image: {e}")
        return None


def generate_images_all_models(prompt: str) -> dict:
    results = {}

    try:
        b64_img = generate_image_base64(prompt)
        results["gemini_base64"] = b64_img
        print("[Gemini] Image generated successfully.")
    except Exception as e:
        results["gemini_base64"] = None
        print(f"[Gemini] Error: {e}")

    image_models = ["flux", "dall-e-3", "dall-e-3"]

    flux_urls = []
    dalle_urls = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_query_image, model, prompt): model for model in image_models}

        for future in as_completed(futures):
            model = futures[future]
            image_url = future.result()
            if not image_url:
                continue

            if model == "flux":
                flux_urls.append(image_url)
            elif model == "dall-e-3":
                dalle_urls.append(image_url)

    results["flux_urls"] = flux_urls
    results["dalle_urls"] = dalle_urls

    return results
