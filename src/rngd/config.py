"""Central configuration: env loading, model routing, paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Groq model routing: cheaper/faster model for straightforward structured
# extraction, larger reasoning model for tasks that trade off multiple
# constraints. This keeps the harness cost/latency-aware rather than sending
# every call to the biggest model available.
GROQ_MODEL_LIGHT = "openai/gpt-oss-20b"
GROQ_MODEL_REASONING = "openai/gpt-oss-120b"

MODEL_ROUTE = {
    "intake": GROQ_MODEL_LIGHT,
    "zoning": GROQ_MODEL_LIGHT,
    "baap_matching": GROQ_MODEL_REASONING,
    "conflict": GROQ_MODEL_REASONING,
    "design_critic": GROQ_MODEL_REASONING,
    "ask_agent": GROQ_MODEL_LIGHT,
}

# Set RNGD_USE_TFIDF=1 to force the lexical (scikit-learn) retriever instead of
# downloading/loading the sentence-transformers embedding model. Useful for
# fast iteration or offline environments; the pipeline behaves identically.
USE_TFIDF = os.environ.get("RNGD_USE_TFIDF", "0") == "1"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
SAMPLE_PROJECTS_DIR = DATA_DIR / "sample_projects"
BAAP_CATALOG_JSON = DATA_DIR / "baap_catalog.json"
OUTPUTS_DIR = ROOT / "outputs"

MAX_SIMULATION_ITERATIONS = 3
