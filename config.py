# config.py

from __future__ import annotations
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ────────────────────────────────────────────────
    anthropic_api_key: str          # required — raises ValidationError if missing
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 8192
    anthropic_temperature: float = 0.1

    # ── GitHub ───────────────────────────────────────────────────
    github_token: str               # required — raises ValidationError if missing

    # ── ArXiv ────────────────────────────────────────────────────
    arxiv_max_results_per_query: int = 50
    arxiv_rate_limit_sleep: float = 1.0

    # ── Pipeline ─────────────────────────────────────────────────
    pipeline_max_papers: int = 100
    pipeline_concurrency: int = 3
    skip_papers_without_repo: bool = True
    claim_min_confidence: float = 0.0

    # ── Discovery loop ───────────────────────────────────────────
    discovery_max_rounds: int = 3
    discovery_min_papers: int = 10

    # ── Storage ──────────────────────────────────────────────────
    db_path: Path = Path("./data/audit.db")
    artifacts_dir: Path = Path("./artifacts")
    report_output_dir: Path = Path("./reports")
    pdf_cache_dir: Path = Path("./artifacts/pdfs")

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Path = Path("./logs/audit.log")

    # ── MCP (optional interoperability layer) ────────────────────
    mcp_auth_token: str = "changeme"
    mcp_rate_limit_per_min: int = 30

    @field_validator(
        "db_path", "artifacts_dir", "report_output_dir", "pdf_cache_dir", "log_file",
        mode="before",
    )
    @classmethod
    def _make_path(cls, v: str | Path) -> Path:
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("anthropic_api_key")
    @classmethod
    def _check_api_key(cls, v: str) -> str:
        if not v or v == "your_key_here":
            raise ValueError("ANTHROPIC_API_KEY must be set in .env")
        return v

    @field_validator("github_token")
    @classmethod
    def _check_github_token(cls, v: str) -> str:
        if not v or v.startswith("ghp_placeholder"):
            raise ValueError("GITHUB_TOKEN must be set in .env")
        return v


settings = Settings()
