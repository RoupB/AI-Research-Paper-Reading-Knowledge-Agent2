# tools/arxiv_tool.py

from __future__ import annotations
import asyncio
import re
from typing import Optional

import arxiv

from agents.base_agent import get_logger, with_retry
from config import settings
from tools import pdf_tool

log = get_logger(__name__)

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _extract_arxiv_id(url_or_id: str) -> str:
    m = _ARXIV_ID_RE.search(url_or_id)
    return m.group(1) if m else url_or_id.rsplit("/", 1)[-1].replace(".pdf", "")


_SORT_MAP = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submittedDate": arxiv.SortCriterion.SubmittedDate,
    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
}


@with_retry(max_attempts=5, backoff_base=2.0, jitter_max=8.0)
async def search_arxiv(
    query: str,
    max_results: int = 50,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "relevance",
) -> list[dict]:
    """
    Search arXiv and return paper metadata dicts.
    Each dict: {arxiv_id, title, authors, abstract, published, pdf_url, arxiv_url}
    """
    full_query = query
    if date_from or date_to:
        lo = (date_from or "1900-01-01").replace("-", "")
        hi = (date_to or "2100-01-01").replace("-", "")
        full_query = f"{query} AND submittedDate:[{lo} TO {hi}]"

    def _run() -> list[dict]:
        client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=settings.arxiv_rate_limit_sleep)
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=_SORT_MAP.get(sort_by, arxiv.SortCriterion.Relevance),
        )
        results: list[dict] = []
        for r in client.results(search):
            results.append(
                {
                    "arxiv_id": _extract_arxiv_id(r.entry_id),
                    "title": r.title.strip().replace("\n", " "),
                    "authors": [a.name for a in r.authors],
                    "abstract": (r.summary or "").strip().replace("\n", " "),
                    "published": r.published.isoformat() if r.published else None,
                    "pdf_url": r.pdf_url,
                    "arxiv_url": r.entry_id,
                }
            )
        return results

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _run)
    await asyncio.sleep(settings.arxiv_rate_limit_sleep)
    log.info("arxiv_search", query=query, found=len(results))
    return results


async def fetch_paper_text(
    pdf_url: str,
    pages: Optional[int] = None,
    full: bool = False,
) -> str:
    """
    Download a PDF and extract text (cached). When `pages` is given, return only
    the first N pages worth of text (approximated by page-break splits).
    """
    arxiv_id = _extract_arxiv_id(pdf_url)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        pdf_tool.fetch_and_cache_pdf,
        pdf_url,
        arxiv_id,
        str(settings.pdf_cache_dir),
    )
    if pages is not None and not full:
        chunks = text.split("\n\n")
        return "\n\n".join(chunks[: max(pages * 3, pages)])
    return text
