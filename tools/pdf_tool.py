# tools/pdf_tool.py
#
# Three SYNCHRONOUS functions. Agents must call these via
# asyncio.run_in_executor(None, fn, *args) — never directly from async code.

from __future__ import annotations
import re
from pathlib import Path

import httpx

from agents.base_agent import get_logger

log = get_logger(__name__)

try:
    import pdfplumber  # noqa: F401
    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:  # pragma: no cover
    _HAS_PYMUPDF = False


_SAFE_ID = re.compile(r"[^A-Za-z0-9.]")


def _safe_arxiv_id(arxiv_id: str) -> str:
    """Sanitise an arxiv id for safe use as a filename (no path traversal)."""
    return _SAFE_ID.sub("_", arxiv_id)


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                txt = ""
            parts.append(txt)
    return "\n\n".join(parts)


def _extract_with_pymupdf(pdf_path: Path) -> str:
    import fitz

    parts: list[str] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page in doc:
            parts.append(page.get_text())
    finally:
        doc.close()
    return "\n\n".join(parts)


def fetch_and_cache_pdf(
    pdf_url: str,
    arxiv_id: str,
    cache_dir: str,
    max_chars: int = 100_000,
) -> str:
    """
    Download a PDF, extract its text, cache to {cache_dir}/{arxiv_id}.txt.
    Returns the extracted (possibly truncated) text string.
    """
    safe_id = _safe_arxiv_id(arxiv_id)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    txt_path = cache / f"{safe_id}.txt"
    pdf_path = cache / f"{safe_id}.pdf"

    # 1. Cache hit
    if txt_path.exists() and txt_path.stat().st_size > 0:
        log.info("cache_hit", arxiv_id=arxiv_id)
        return txt_path.read_text(encoding="utf-8")

    # 2. Download
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(pdf_url)
            if resp.status_code != 200:
                raise RuntimeError(str(resp.status_code))
            content = resp.content
    except httpx.HTTPError as exc:
        raise RuntimeError(str(exc)) from exc

    if not content:
        log.warning("empty_pdf", arxiv_id=arxiv_id)
        return ""
    pdf_path.write_bytes(content)

    # 3/4. Extract text (pdfplumber, fall back to pymupdf)
    text = ""
    if _HAS_PDFPLUMBER:
        try:
            text = _extract_with_pdfplumber(pdf_path)
        except Exception as exc:  # noqa: BLE001
            if "encrypt" in str(exc).lower() or "password" in str(exc).lower():
                log.warning("encrypted_pdf", arxiv_id=arxiv_id)
                pdf_path.unlink(missing_ok=True)
                return ""
            text = ""
    if len(text) < 100 and _HAS_PYMUPDF:
        try:
            text = _extract_with_pymupdf(pdf_path)
        except Exception:  # noqa: BLE001
            text = text

    # 5. Both failed
    if len(text) < 100:
        log.error("pdf_extraction_failed", arxiv_id=arxiv_id)
        pdf_path.unlink(missing_ok=True)
        return ""

    # 6. Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[TRUNCATED AT {max_chars} CHARS]"
        log.warning("pdf_truncated", arxiv_id=arxiv_id, max_chars=max_chars)

    # 7/8. Save text, delete pdf
    txt_path.write_text(text, encoding="utf-8")
    pdf_path.unlink(missing_ok=True)

    # 9. Return
    return text


def parse_tables(pdf_path: str, max_tables: int = 50) -> list[dict]:
    """
    Extract tables from a PDF. Returns a list of
    {"page", "table_index", "headers", "rows", "raw_text"}.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)

    tables: list[dict] = []
    if _HAS_PDFPLUMBER:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages):
                try:
                    extracted = page.extract_tables() or []
                except Exception:  # noqa: BLE001
                    log.warning("table_page_skipped", page=page_no)
                    continue
                for t_idx, raw in enumerate(extracted):
                    if not raw:
                        continue
                    first = [c or "" for c in raw[0]]
                    header_is_text = any(
                        c and not _is_number(c) for c in first
                    )
                    if header_is_text:
                        headers = [c.strip() or f"col_{i}" for i, c in enumerate(first)]
                        body = raw[1:]
                    else:
                        headers = [f"col_{i}" for i in range(len(first))]
                        body = raw
                    rows = [
                        {headers[i]: (cell or "") for i, cell in enumerate(r) if i < len(headers)}
                        for r in body
                    ]
                    tables.append(
                        {
                            "page": page_no,
                            "table_index": t_idx,
                            "headers": headers,
                            "rows": rows,
                            "raw_text": "\n".join(
                                "\t".join((c or "") for c in r) for r in raw
                            ),
                        }
                    )
                    if len(tables) >= max_tables:
                        log.warning("max_tables_exceeded", max_tables=max_tables)
                        return tables

    if not tables:
        # Fallback: regex over text-like tabular lines
        tables = _regex_tables(path, max_tables)

    return tables


def _is_number(s: str) -> bool:
    try:
        float(s.replace("%", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def _regex_tables(path: Path, max_tables: int) -> list[dict]:
    text = ""
    if _HAS_PDFPLUMBER:
        try:
            text = _extract_with_pdfplumber(path)
        except Exception:  # noqa: BLE001
            text = ""
    if not text and _HAS_PYMUPDF:
        try:
            text = _extract_with_pymupdf(path)
        except Exception:  # noqa: BLE001
            text = ""

    rows: list[dict] = []
    for line in text.splitlines():
        cells = re.split(r"\t|\s*\|\s*", line.strip())
        cells = [c for c in cells if c]
        if len(cells) >= 3:
            rows.append({f"col_{i}": c for i, c in enumerate(cells)})
    if not rows:
        return []
    return [
        {
            "page": -1,
            "table_index": 0,
            "headers": list(rows[0].keys()),
            "rows": rows[:200],
            "raw_text": text[:5000],
        }
    ]


_SECTION_PATTERN = re.compile(
    r"^(\d+\.?\s*)?(abstract|introduction|related work|background|method|approach|"
    r"model|experiment|result|evaluation|discussion|conclusion|appendix|limitation)",
    re.IGNORECASE | re.MULTILINE,
)

_CANONICAL = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "related_work",
    "method": "method",
    "approach": "method",
    "model": "method",
    "experiment": "experiments",
    "result": "experiments",
    "evaluation": "experiments",
    "discussion": "experiments",
    "conclusion": "conclusion",
    "limitation": "conclusion",
    "appendix": "appendix",
}

_KEYS = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "conclusion",
    "appendix",
]


def extract_sections(text: str, max_section_chars: int = 20_000) -> dict[str, str]:
    """Split paper text into 7 canonical sections. Always returns all 7 keys."""
    out: dict[str, str] = {k: "" for k in _KEYS}
    if not text:
        return out

    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        log.warning("no_sections_detected")
        out["method"] = text[:max_section_chars]
        if len(text) > max_section_chars:
            out["method"] += "[TRUNCATED]"
        return out

    for i, m in enumerate(matches):
        name = m.group(2).lower()
        canonical = _CANONICAL.get(name, "method")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        out[canonical] = (out[canonical] + "\n" + body).strip() if out[canonical] else body

    for k in _KEYS:
        if len(out[k]) > max_section_chars:
            out[k] = out[k][:max_section_chars] + "[TRUNCATED]"

    return out
