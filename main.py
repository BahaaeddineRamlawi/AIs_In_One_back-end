from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import warnings

from logger import logger
from llm import get_model_chains
from image_gen import generate_image_base64

from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from keep_alive import keep_alive
keep_alive()

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
        image_b64 = generate_image_base64(prompt)
        image_data_uri = f"data:image/png;base64,{image_b64}"
        return {"imageUrl": image_data_uri}
    except Exception as e:
        return {"error": str(e)}
    
    # xhQX8dESAeSInIb7LVemOB3KxmjIcQuJYUxHhEjJWlRwug3pS9m6c6VDXfLk