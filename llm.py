import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

from utils.key_rotator import PersistentKeyRotator

load_dotenv()

INDEX_FILE = "key_index.json"

gemini_rotator = PersistentKeyRotator(
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    index_file=INDEX_FILE,
)

openrouter_rotator = PersistentKeyRotator(
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2"),
    index_file=INDEX_FILE,
)

mistral_rotator = PersistentKeyRotator(
    os.getenv("MISTRAL_API_KEY_1"),
    os.getenv("MISTRAL_API_KEY_2"),
    index_file=INDEX_FILE,
)

together_rotator = PersistentKeyRotator(
    os.getenv("TOGETHER_API_KEY_1"),
    os.getenv("TOGETHER_API_KEY_2"),
    index_file=INDEX_FILE,
)

cohere_rotator = PersistentKeyRotator(
    os.getenv("COHERE_API_KEY_1"),
    os.getenv("COHERE_API_KEY_2"),
    index_file=INDEX_FILE,
)

__all__ = [
    "get_model_chains",
    "gemini_rotator",
    "mistral_rotator",
    "together_rotator",
    "openrouter_rotator",
    "cohere_rotator",
]


def get_model_chains():
    current_index = gemini_rotator.index
    gemini_key = gemini_rotator.keys[current_index]
    openrouter_key = openrouter_rotator.keys[current_index]
    mistral_key = mistral_rotator.keys[current_index]
    together_key = together_rotator.keys[current_index]
    cohere_key = cohere_rotator.keys[current_index]

    gemini = ConversationChain(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_key,
            temperature=0.7,
        ),
        memory=ConversationBufferMemory(),
    )

    mistral = ConversationChain(
        llm=ChatMistralAI(
            api_key=mistral_key,
            model="mistral-tiny",
            temperature=0.7,
        ),
        memory=ConversationBufferMemory(),
    )

    mixtral = ConversationChain(
        llm=ChatOpenAI(
            temperature=0.7,
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            openai_api_key=together_key,
            base_url="https://api.together.xyz/v1",
        ),
        memory=ConversationBufferMemory(),
    )

    llama = ConversationChain(
        llm=ChatOpenAI(
            temperature=0.7,
            model_name="meta-llama/llama-3-8b-instruct",
            openai_api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        ),
        memory=ConversationBufferMemory(),
    )

    cohere = ConversationChain(
        llm=ChatCohere(
            cohere_api_key=cohere_key,
            model="command-r-plus",
            temperature=0.7,
        ),
        memory=ConversationBufferMemory(),
    )

    deepseek = ConversationChain(
        llm=ChatOpenAI(
            temperature=0.7,
            model_name="deepseek/deepseek-chat-v3-0324:free",
            openai_api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        ),
        memory=ConversationBufferMemory(),
    )


    gemini_rotator.rotate_key()

    return {
        "Gemini": gemini,
        "Mistral": mistral,
        "Mixtral": mixtral,
        "LLaMA 3": llama,
        "Cohere": cohere,
        "DeepSeek": deepseek,
    }
