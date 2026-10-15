"""Central configuration: env loading, model routing, paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

# All LLM calls go through OpenRouter. Only non-Chinese providers are used;
# llm.assert_allowed() rejects anything else before a request is sent.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Light model for structured extraction, reasoning model for multi-constraint
# judgment, and one expensive arbiter model for the final synthesis call.
LLM_LIGHT = os.environ.get("OPENROUTER_LIGHT_MODEL", "openai/gpt-oss-20b")
LLM_REASONING = os.environ.get("OPENROUTER_REASONING_MODEL", "openai/gpt-oss-120b")
ARBITER_MODEL = os.environ.get("OPENROUTER_ARBITER_MODEL", "anthropic/claude-sonnet-5")
# Writer of the final report prose (optimizer AI report). A stronger writing model than the
# agents use: the call is one per report, so quality matters more than cost.
REPORT_MODEL = os.environ.get("OPENROUTER_REPORT_MODEL", "anthropic/claude-sonnet-5.5")

MODEL_ROUTE = {
    "intake": LLM_LIGHT,
    "zoning": LLM_LIGHT,
    "baap_matching": LLM_REASONING,
    "conflict": LLM_REASONING,
    "design_critic": LLM_REASONING,
    "ask_agent": LLM_LIGHT,
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
