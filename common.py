"""Shared config used by ingest.py and app.py."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "question_papers"
PARSED_DIR = ROOT / "data" / "parsed"   # cached LLM transcriptions (one JSON per paper)
INDEX_DIR = ROOT / "data" / "faiss_index"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def get_embeddings():
    # Small local model: free, no API calls, ~130MB download on first run.
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_llm(temperature: float = 0):
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
