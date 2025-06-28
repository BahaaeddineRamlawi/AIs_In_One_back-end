from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from logger import logger
from llm import get_model_chains

import warnings
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
