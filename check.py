import os
import time
import json
from typing import TypedDict, List

from dotenv import load_dotenv

load_dotenv()  # charge ./.env pour que `uv run check.py` voie les clés

# --- API keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "VOTRE_CLE_GROQ_ICI")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "VOTRE_CLE_TAVILY_ICI")
GROQ_MODEL = "openai/gpt-oss-20b"   # llama-3.3-70b-versatile renvoie 404 sur ce compte

GROQ_OK = GROQ_API_KEY.startswith("gsk_") and len(GROQ_API_KEY) > 30
TAVILY_OK = TAVILY_API_KEY.startswith("tvly-") and len(TAVILY_API_KEY) > 20

print(f"Groq   : {'✅ ok' if GROQ_OK else '❌ clé manquante'}")
print(f"Tavily : {'✅ ok' if TAVILY_OK else '❌ clé manquante (web search désactivée)'}")
print(f"Modèle : {GROQ_MODEL}")