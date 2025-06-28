from dotenv import load_dotenv
import os
from utils.key_rotator import PersistentKeyRotator
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

load_dotenv()

rotator = PersistentKeyRotator(
    os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")
)

def get_model_chains():
    index = rotator.get_next_key_index()

    gemini_key = os.getenv(f"GEMINI_API_KEY_{index + 1}")
    openrouter_key = os.getenv(f"OPENROUTER_API_KEY_{index + 1}")
    mistral_key = os.getenv(f"MISTRAL_API_KEY_{index + 1}")
    together_key = os.getenv(f"TOGETHER_API_KEY_{index + 1}")
    cohere_key = os.getenv(f"COHERE_API_KEY_{index + 1}")

    return {
        "Gemini": ConversationChain(
            llm=ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=gemini_key,
                temperature=0.7,
            ),
            memory=ConversationBufferMemory(),
        ),
        "Mistral": ConversationChain(
            llm=ChatMistralAI(
                api_key=mistral_key,
                model="mistral-tiny",
                temperature=0.7,
            ),
            memory=ConversationBufferMemory(),
        ),
        "Mixtral": ConversationChain(
            llm=ChatOpenAI(
                temperature=0.7,
                model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                openai_api_key=together_key,
                base_url="https://api.together.xyz/v1",
            ),
            memory=ConversationBufferMemory(),
        ),
        "LLaMA 3": ConversationChain(
            llm=ChatOpenAI(
                temperature=0.7,
                model_name="meta-llama/llama-3-8b-instruct",
                openai_api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            ),
            memory=ConversationBufferMemory(),
        ),
        "Cohere": ConversationChain(
            llm=ChatCohere(
                cohere_api_key=cohere_key,
                model="command-r-plus",
                temperature=0.7,
            ),
            memory=ConversationBufferMemory(),
        ),
    }
