# tools/github_tool.py

from __future__ import annotations
import asyncio
import re
from typing import Optional

import httpx

from agents.base_agent import get_logger, with_retry
from config import settings

log = get_logger(__name__)


class ConfigError(RuntimeError):
    """Raised when GITHUB_TOKEN is missing."""


_GITHUB_URL_RE = re.compile(r"github\.com[/:]([\w.-]+)/([\w.-]+)", re.IGNORECASE)

_PRIORITY_PREFIXES = ("train", "run", "finetune", "fine_tune", "main")
_PRIORITY_NAMES = ("readme.md", "config.yaml", "config.yml")


def normalize_repo_url(url: str) -> Optional[str]:
    """Normalise any GitHub URL to https://github.com/{owner}/{repo}."""
    if not url:
        return None
    m = _GITHUB_URL_RE.search(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    repo = repo.replace(".git", "").split("?")[0].split("#")[0].rstrip("/")
    return f"https://github.com/{owner}/{repo}"


def _owner_repo(url: str) -> Optional[tuple[str, str]]:
    norm = normalize_repo_url(url)
    if not norm:
        return None
    _, _, tail = norm.partition("github.com/")
    owner, _, repo = tail.partition("/")
    return (owner, repo) if owner and repo else None


def _require_token() -> str:
    if not settings.github_token:
        raise ConfigError("GITHUB_TOKEN must be set in .env")
    return settings.github_token


@with_retry(max_attempts=3, backoff_base=2.0, jitter_max=2.0, retriable=(httpx.HTTPError,))
async def resolve_repo(
    candidate_url: Optional[str],
    paper_title: str,
    authors: list[str],
) -> tuple[Optional[str], float]:
    """
    Verify a candidate GitHub URL exists. If None, attempt GitHub search.
    Returns (verified_url, confidence_score).
    """
    token = _require_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        # 1. Verify candidate (try parent levels on 404)
        if candidate_url:
            norm = normalize_repo_url(candidate_url)
            pair = _owner_repo(norm) if norm else None
            if pair:
                owner, repo = pair
                meta = await _get_repo_meta(client, owner, repo)
                if meta is not None:
                    conf = 0.95 if not meta.get("fork") else 0.6
                    return norm, conf

        # 2. Search GitHub by title
        query = re.sub(r"[^\w\s]", " ", paper_title).strip()
        try:
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": 5},
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    top = items[0]
                    norm = normalize_repo_url(top["html_url"])
                    conf = 0.55 if top.get("stargazers_count", 0) > 0 else 0.45
                    return norm, conf
        except httpx.HTTPError as exc:
            log.warning("github_search_failed", error=str(exc))

    return None, 0.0


async def _get_repo_meta(client: httpx.AsyncClient, owner: str, repo: str) -> Optional[dict]:
    resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
    if resp.status_code == 200:
        return resp.json()
    return None


def _relevance_rank(path: str) -> int:
    name = path.lower().rsplit("/", 1)[-1]
    if name in _PRIORITY_NAMES:
        return 1
    if any(name.startswith(p) for p in _PRIORITY_PREFIXES):
        return 0
    if name.endswith((".sh", ".yaml", ".yml")):
        return 2
    return 3


@with_retry(max_attempts=3, backoff_base=2.0, jitter_max=2.0, retriable=(httpx.HTTPError,))
async def fetch_repo_tree(
    repo_url: str,
    max_files: int = 200,
    extensions: list[str] | None = None,
) -> list[dict]:
    """Return [{path, size, download_url}] for matching files, sorted by relevance."""
    if extensions is None:
        extensions = [".py", ".sh", ".yaml", ".yml", ".json", ".md"]
    token = _require_token()
    pair = _owner_repo(repo_url)
    if not pair:
        return []
    owner, repo = pair
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        meta = await _get_repo_meta(client, owner, repo)
        default_branch = meta.get("default_branch", "main") if meta else "main"
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if resp.status_code != 200:
            log.warning("repo_tree_failed", repo=repo_url, status=resp.status_code)
            return []
        tree = resp.json().get("tree", [])

    files: list[dict] = []
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = node["path"]
        if any(seg in path for seg in ("__pycache__", ".git/")):
            continue
        if not any(path.lower().endswith(ext) for ext in extensions):
            continue
        size = node.get("size", 0)
        if size > 500_000:
            continue
        files.append(
            {
                "path": path,
                "size": size,
                "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{path}",
            }
        )

    files.sort(key=lambda f: _relevance_rank(f["path"]))
    return files[:max_files]


@with_retry(max_attempts=3, backoff_base=2.0, jitter_max=2.0, retriable=(httpx.HTTPError,))
async def fetch_file(download_url: str) -> str:
    """Fetch raw file content from GitHub. Returns '' on decode error."""
    token = _require_token()
    headers = {"Authorization": f"Bearer {token}"}
    await asyncio.sleep(1.0)  # secondary rate-limit courtesy
    async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = await client.get(download_url)
        if resp.status_code != 200:
            log.warning("file_fetch_failed", url=download_url, status=resp.status_code)
            return ""
        try:
            return resp.content.decode("utf-8")
        except UnicodeDecodeError:
            log.warning("file_decode_error", url=download_url)
            return ""
