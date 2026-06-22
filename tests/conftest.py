# tests/conftest.py
#
# Shared pytest fixtures used by all test modules.

from __future__ import annotations
import json
import sys
from itertools import cycle
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Ensure project root importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from db import init_db  # noqa: E402
from models import BenchmarkClaim, CodeFact, Paper, PaperStatus  # noqa: E402


def _message(text: str) -> MagicMock:
    msg = MagicMock()
    block = MagicMock()
    block.text = text
    msg.content = [block]
    return msg


def make_mock_client(responses) -> AsyncMock:
    """
    Build an AsyncMock Anthropic client.
    `responses` may be a single dict/str or a list of dict/str cycled per call.
    """
    if not isinstance(responses, list):
        responses = [responses]
    texts = [r if isinstance(r, str) else json.dumps(r) for r in responses]
    pool = cycle(texts)
    client = AsyncMock()

    async def _create(*_args, **_kwargs):
        return _message(next(pool))

    client.messages.create = AsyncMock(side_effect=_create)
    return client


# ── Temporary database ────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def test_db(tmp_path, monkeypatch):
    """Create a fresh empty database for each test (monkeypatches settings.db_path)."""
    db_path = tmp_path / "test_audit.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "report_output_dir", tmp_path / "reports")
    await init_db.create_tables(db_path)
    yield db_path


# ── Mock Anthropic client factory ─────────────────────────────────────────────
@pytest.fixture
def mock_llm():
    """Factory: mock_llm({...}) or mock_llm([{...}, {...}]) → AsyncMock client."""
    return make_mock_client


# ── Sample fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2305.14314",
        title="QLoRA: Efficient Finetuning of Quantized LLMs",
        authors=["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
        abstract="We present QLoRA, an efficient finetuning approach that reduces memory usage.",
        published="2023-05-23T00:00:00+00:00",
        pdf_url="https://arxiv.org/pdf/2305.14314",
        arxiv_url="https://arxiv.org/abs/2305.14314",
        lora_variant_tag="QLoRA",
        status=PaperStatus.DISCOVERED,
    )


@pytest.fixture
def sample_claim() -> BenchmarkClaim:
    return BenchmarkClaim(
        paper_id="2305.14314",
        claim_id="claim-1",
        metric="accuracy",
        dataset="GLUE/MNLI",
        model_base="LLaMA-7B",
        reported_value=90.2,
        unit="%",
        conditions={"rank": "8"},
        is_conditional=True,
        source_section="Table 2",
        raw_text="QLoRA achieves 90.2% on MNLI with rank 8.",
    )


@pytest.fixture
def sample_fact() -> CodeFact:
    return CodeFact(
        paper_id="2305.14314",
        repo_url="https://github.com/artidoro/qlora",
        fact_id="fact-1",
        fact_type="hyperparameter",
        key="rank",
        value="16",
        file_path="train.py",
        line_range=(10, 12),
        evidence="parser.add_argument('--rank', default=16)",
    )
