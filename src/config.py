"""Central settings + shared constants.

All configuration is env-driven (see env.example). Retrieval top-k values and
the legal disclaimer live here too, per CLAUDE.md ("not scattered").
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider
    llm_provider: str = "openai_compatible"  # ollama | anthropic | openai_compatible | gemini

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = "https://api.groq.com/openai/v1"
    openai_compatible_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Qdrant. Local docker compose by default; if qdrant_url is empty and
    # qdrant_path is set, retrieval/ingest fall back to embedded local mode.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_path: str = ""

    # API auth — fail-closed if unset (checked at request time, not import time).
    rag_api_key: str = ""

    # Retrieval
    rerank_enabled: bool = False
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 5
    top_k_rerank_candidates: int = 50

    # Contract corpus (M5)
    contract_ttl_hours: int = 24
    contract_max_mb: int = 5
    contract_max_pages: int = 20
    top_k_contract: int = 3

    # Contract analysis (M6) — law articles retrieved per clause. Tighter than
    # top_k_final: /analyze makes one LLM call per clause, so context stays small.
    top_k_analyze: int = 3
    # Pause between clause LLM calls; raise (e.g. 3–5 s) if the provider's
    # per-minute token window keeps 429-ing mid-report. 0 = off.
    analyze_clause_delay_seconds: float = 0.0


settings = Settings()

# Embedding model — the SAME model embeds documents and queries (see ARCHITECTURE §2.1).
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Cross-encoder reranker (M4, flag-gated via RERANK_ENABLED — ARCHITECTURE §2.4).
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Qdrant collections / named vectors
LAW_COLLECTION = "law"
CONTRACTS_COLLECTION = "contracts"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

DISCLAIMER = "Bu bilgilendirme amaçlıdır, hukuki tavsiye değildir."
