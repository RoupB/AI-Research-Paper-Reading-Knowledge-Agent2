# tests/test_arxiv_tool.py

from __future__ import annotations

from tools import arxiv_tool


def test_extract_arxiv_id_from_pdf_url():
    assert arxiv_tool._extract_arxiv_id("https://arxiv.org/pdf/2305.14314") == "2305.14314"


def test_extract_arxiv_id_from_abs_url():
    assert arxiv_tool._extract_arxiv_id("http://arxiv.org/abs/2106.09685v2") == "2106.09685"


def test_extract_arxiv_id_with_version():
    assert arxiv_tool._extract_arxiv_id("2402.09353v1") == "2402.09353"


def test_extract_arxiv_id_five_digit():
    assert arxiv_tool._extract_arxiv_id("https://arxiv.org/pdf/2310.11454") == "2310.11454"


def test_sort_map_has_relevance():
    assert "relevance" in arxiv_tool._SORT_MAP
    assert "submittedDate" in arxiv_tool._SORT_MAP
