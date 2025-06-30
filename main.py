from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import warnings

from logger import logger
from llm import get_model_chains
from image_gen import generate_images_all_models

from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    logger.info(f"Received message: {message}")

    try:
        model_chains = get_model_chains()
        tasks = [chain.ainvoke(message) for chain in model_chains.values()]
        results = await asyncio.gather(*tasks)

        replies = [
            {"model": model_name, "text": result}
            for model_name, result in zip(model_chains.keys(), results)
        ]

        logger.info(f"Replies: {replies}")
        return {"replies": replies}

    except Exception as e:
        logger.error(f"Error during chat endpoint: {e}", exc_info=True)
        return {"replies": [{"model": "Error", "text": str(e)}]}


@app.post("/api/generate-image")
async def generate_image_endpoint(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return {"error": "Missing prompt"}

    try:
        images = generate_images_all_models(prompt)

        gemini_data_uri = None
        if images.get("gemini_base64"):
            gemini_data_uri = f"data:image/png;base64,{images['gemini_base64']}"

        return {
            "images": {
                "gemini_base64": gemini_data_uri,
                "dalle_urls": images.get("dalle_urls", []),
                "imagen_urls": images.get("imagen_urls", []),
                "flux_urls": images.get("flux_urls", []),
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/wake")
async def wake_server():
    return {"status": "awake"}
