# tests/test_github_tool.py

from __future__ import annotations

from tools import github_tool


def test_normalize_strips_git_suffix():
    assert (
        github_tool.normalize_repo_url("https://github.com/artidoro/qlora.git")
        == "https://github.com/artidoro/qlora"
    )


def test_normalize_strips_query_and_trailing():
    assert (
        github_tool.normalize_repo_url("https://github.com/owner/repo/?tab=readme")
        == "https://github.com/owner/repo"
    )


def test_normalize_handles_subpath():
    assert (
        github_tool.normalize_repo_url("github.com/owner/repo/tree/main/src")
        == "https://github.com/owner/repo"
    )


def test_normalize_returns_none_for_non_github():
    assert github_tool.normalize_repo_url("https://example.com/foo") is None
    assert github_tool.normalize_repo_url("") is None


def test_owner_repo():
    assert github_tool._owner_repo("https://github.com/a/b") == ("a", "b")
    assert github_tool._owner_repo("not a url") is None


def test_relevance_rank_orders_training_files_first():
    ranks = {
        "train.py": github_tool._relevance_rank("train.py"),
        "src/finetune_lora.py": github_tool._relevance_rank("src/finetune_lora.py"),
        "README.md": github_tool._relevance_rank("README.md"),
        "run.sh": github_tool._relevance_rank("scripts/run.sh"),
        "utils.py": github_tool._relevance_rank("utils.py"),
    }
    # training/run files outrank generic utils
    assert ranks["train.py"] < ranks["utils.py"]
    assert ranks["src/finetune_lora.py"] < ranks["utils.py"]
    assert ranks["README.md"] < ranks["utils.py"]
