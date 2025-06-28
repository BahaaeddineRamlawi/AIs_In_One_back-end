import os
from dotenv import load_dotenv
from utils.key_rotator import PersistentKeyRotator

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

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


def get_model_chains():
    gemini = ConversationChain(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_rotator.get_next_key(),
            temperature=0.7,
        ),
        memory=ConversationBufferMemory(),
    )

    mistral = ConversationChain(
        llm=ChatMistralAI(
            api_key=mistral_rotator.get_next_key(),
            model="mistral-tiny",
            temperature=0.7,
        ),
        memory=ConversationBufferMemory(),
    )

    mixtral = ConversationChain(
        llm=ChatOpenAI(
            temperature=0.7,
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            openai_api_key=together_rotator.get_next_key(),
            base_url="https://api.together.xyz/v1",
        ),
        memory=ConversationBufferMemory(),
    )

    llama = ConversationChain(
        llm=ChatOpenAI(
            temperature=0.7,
            model_name="meta-llama/llama-3-8b-instruct",
            openai_api_key=openrouter_rotator.get_next_key(),
            base_url="https://openrouter.ai/api/v1",
        ),
        memory=ConversationBufferMemory(),
    )

    cohere = ConversationChain(
        llm=ChatCohere(
            cohere_api_key=cohere_rotator.get_next_key(),
            model="command-r-plus",
            temperature=0.7,
        ),
        memory=ConversationBufferMemory(),
    )

    return {
        "Gemini": gemini,
        "Mistral": mistral,
        "Mixtral": mixtral,
        "LLaMA 3": llama,
        "Cohere": cohere,
    }
