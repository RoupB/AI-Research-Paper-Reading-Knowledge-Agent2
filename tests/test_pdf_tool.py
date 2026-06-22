# tests/test_pdf_tool.py

from __future__ import annotations

import pytest

from tools import pdf_tool


def test_extract_sections_empty():
    out = pdf_tool.extract_sections("")
    assert set(out.keys()) == {
        "abstract", "introduction", "related_work", "method",
        "experiments", "conclusion", "appendix",
    }
    assert all(v == "" for v in out.values())


def test_extract_sections_basic():
    text = (
        "Abstract\nWe propose a method.\n"
        "1. Introduction\nLoRA is great.\n"
        "3. Experiments\nWe report 90% accuracy.\n"
        "Conclusion\nDone."
    )
    out = pdf_tool.extract_sections(text)
    assert "We propose a method" in out["abstract"]
    assert "LoRA is great" in out["introduction"]
    assert "90% accuracy" in out["experiments"]
    assert "Done" in out["conclusion"]


def test_extract_sections_merges_results_into_experiments():
    text = "Results\nTable shows 88.\nEvaluation\nWe used MNLI."
    out = pdf_tool.extract_sections(text)
    assert "Table shows 88" in out["experiments"]
    assert "We used MNLI" in out["experiments"]


def test_extract_sections_no_headers_goes_to_method():
    text = "Just some free text with no recognizable section headers here."
    out = pdf_tool.extract_sections(text)
    assert out["method"].startswith("Just some free text")


def test_extract_sections_truncates():
    big = "Method\n" + ("x" * 50_000)
    out = pdf_tool.extract_sections(big, max_section_chars=1000)
    assert out["method"].endswith("[TRUNCATED]")
    assert len(out["method"]) <= 1000 + len("[TRUNCATED]")


def test_safe_arxiv_id():
    assert pdf_tool._safe_arxiv_id("2305.14314") == "2305.14314"
    assert "/" not in pdf_tool._safe_arxiv_id("../../etc/passwd")
    assert ".." in pdf_tool._safe_arxiv_id("..")  # dots allowed but no slash
    assert "/" not in pdf_tool._safe_arxiv_id("a/b/c")


def test_parse_tables_missing_file():
    with pytest.raises(FileNotFoundError):
        pdf_tool.parse_tables("does_not_exist_12345.pdf")


def test_is_number():
    assert pdf_tool._is_number("90.2")
    assert pdf_tool._is_number("88%")
    assert not pdf_tool._is_number("accuracy")
