
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Project directories
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ChromaDB collection
COLLECTION_NAME = "mini_rag_documents"
