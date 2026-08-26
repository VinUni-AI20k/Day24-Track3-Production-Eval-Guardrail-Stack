"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- LLM Provider (Day 18 compatibility) ---
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-20b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "model": "gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
if LLM_PROVIDER not in PROVIDERS:
    raise ValueError(
        f"LLM_PROVIDER={LLM_PROVIDER!r} không hợp lệ. "
        f"Chọn một trong: {', '.join(PROVIDERS)}"
    )

_p = PROVIDERS[LLM_PROVIDER]
LLM_API_KEY = os.getenv(_p["key_env"], "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or _p["base_url"]
LLM_MODEL = os.getenv("LLM_MODEL", _p["model"])
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
ENRICH_WITH_LLM = os.getenv("ENRICH_WITH_LLM", "1") not in ("0", "false", "False")


def get_llm_client():
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                  max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT)

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = "gpt-4o-mini"

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
