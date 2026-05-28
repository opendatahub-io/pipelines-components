"""Unit tests for check_container_build_matrix."""

import textwrap
from pathlib import Path

from ..check_container_build_matrix import (
    IGNORE_FILENAME,
    check,
    discover_container_files,
    load_ignore_list,
    parse_matrix_contexts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def _make_workflow(tmp_path: Path, contexts: list[str]) -> Path:
    includes = "\n".join(f"        - name: {ctx}\n          context: {ctx}" for ctx in contexts)
    content = f"""\
jobs:
  build:
    strategy:
      matrix:
        include:
{includes}
"""
    return _write(tmp_path / ".github/workflows/container-build.yml", content)


# ---------------------------------------------------------------------------
# discover_container_files
# ---------------------------------------------------------------------------


class TestDiscoverContainerFiles:
    """Tests for discover_container_files."""

    def test_finds_containerfile(self, tmp_path):
        """Finds a Containerfile under a search root."""
        _write(tmp_path / "components/cat/comp/Containerfile", "FROM python:3.11")
        results = discover_container_files(tmp_path, ["components"])
        assert any("Containerfile" in str(r) for r in results)

    def test_finds_dockerfile(self, tmp_path):
        """Finds a Dockerfile under a search root."""
        _write(tmp_path / "components/cat/comp/Dockerfile", "FROM python:3.11")
        results = discover_container_files(tmp_path, ["components"])
        assert any("Dockerfile" in str(r) for r in results)

    def test_finds_nested_subcategory(self, tmp_path):
        """Finds a Containerfile nested under a subcategory."""
        _write(tmp_path / "components/cat/subcat/comp/Containerfile", "FROM python:3.11")
        results = discover_container_files(tmp_path, ["components"])
        assert len(results) == 1

    def test_missing_root_is_skipped(self, tmp_path):
        """Returns empty list when search root does not exist."""
        results = discover_container_files(tmp_path, ["nonexistent"])
        assert results == []

    def test_multiple_roots(self, tmp_path):
        """Finds container files across multiple search roots."""
        _write(tmp_path / "components/cat/a/Containerfile", "FROM a")
        _write(tmp_path / "pipelines/cat/b/Containerfile", "FROM b")
        results = discover_container_files(tmp_path, ["components", "pipelines"])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# parse_matrix_contexts
# ---------------------------------------------------------------------------


class TestParseMatrixContexts:
    """Tests for parse_matrix_contexts."""

    def test_extracts_contexts(self, tmp_path):
        """Extracts context paths from the matrix workflow file."""
        wf = _make_workflow(tmp_path, ["components/cat/comp", "pipelines/cat/pipe"])
        contexts = parse_matrix_contexts(wf)
        contexts_normalized = {c.replace("\\", "/") for c in contexts}
        assert "components/cat/comp" in contexts_normalized
        assert "pipelines/cat/pipe" in contexts_normalized

    def test_empty_matrix(self, tmp_path):
        """Returns empty set when matrix include list is empty."""
        wf = _write(
            tmp_path / ".github/workflows/container-build.yml",
            """\
            jobs:
              build:
                strategy:
                  matrix:
                    include: []
            """,
        )
        assert parse_matrix_contexts(wf) == set()

    def test_missing_matrix_key(self, tmp_path):
        """Returns empty set when matrix key is missing."""
        wf = _write(
            tmp_path / ".github/workflows/container-build.yml",
            "jobs:\n  build:\n    steps: []\n",
        )
        assert parse_matrix_contexts(wf) == set()

    def test_invalid_yaml_returns_empty(self, tmp_path):
        """Returns empty set when YAML is malformed."""
        wf = _write(tmp_path / ".github/workflows/container-build.yml", ": : :\n  - [")
        assert parse_matrix_contexts(wf) == set()

    def test_null_yaml_returns_empty(self, tmp_path):
        """Returns empty set when YAML file is empty."""
        wf = _write(tmp_path / ".github/workflows/container-build.yml", "")
        assert parse_matrix_contexts(wf) == set()

    def test_non_dict_include_entry_skipped(self, tmp_path):
        """Skips non-dict entries in the include list."""
        wf = _write(
            tmp_path / ".github/workflows/container-build.yml",
            """\
jobs:
  build:
    strategy:
      matrix:
        include:
          - null
          - name: valid
            context: components/valid
""",
        )
        contexts = parse_matrix_contexts(wf)
        contexts_normalized = {c.replace("\\", "/") for c in contexts}
        assert len(contexts_normalized) == 1
        assert "components/valid" in contexts_normalized


# ---------------------------------------------------------------------------
# load_ignore_list
# ---------------------------------------------------------------------------


class TestLoadIgnoreList:
    """Tests for load_ignore_list."""

    def test_loads_paths(self, tmp_path):
        """Loads paths from the ignore file."""
        (tmp_path / IGNORE_FILENAME).write_text("components/foo/bar\npipelines/baz\n")
        result = load_ignore_list(tmp_path)
        result_normalized = {p.replace("\\", "/") for p in result}
        assert "components/foo/bar" in result_normalized
        assert "pipelines/baz" in result_normalized

    def test_ignores_comments_and_blank_lines(self, tmp_path):
        """Skips comments and blank lines in the ignore file."""
        (tmp_path / IGNORE_FILENAME).write_text("# comment\n\ncomponents/foo\n")
        result = load_ignore_list(tmp_path)
        result_normalized = {p.replace("\\", "/") for p in result}
        assert result_normalized == {"components/foo"}

    def test_missing_file_returns_empty(self, tmp_path):
        """Returns empty set when ignore file does not exist."""
        assert load_ignore_list(tmp_path) == set()


# ---------------------------------------------------------------------------
# check (integration)
# ---------------------------------------------------------------------------


class TestCheck:
    """Integration tests for check."""

    def test_all_matched(self, tmp_path):
        """Passes when all container files have a matrix entry."""
        _write(tmp_path / "components/cat/comp/Containerfile", "FROM python:3.11")
        wf = _make_workflow(tmp_path, ["components/cat/comp"])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert all_matched
        assert results[0]["status"] == "ok"

    def test_unmatched_fails(self, tmp_path):
        """Fails when a container file has no matrix entry."""
        _write(tmp_path / "components/cat/comp/Containerfile", "FROM python:3.11")
        wf = _make_workflow(tmp_path, [])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert not all_matched
        assert results[0]["status"] == "unmatched"
        assert "suggestion" in results[0]

    def test_suggestion_contains_context(self, tmp_path):
        """Suggestion includes the correct context path and proper indentation."""
        _write(tmp_path / "components/cat/comp/Containerfile", "FROM python:3.11")
        wf = _make_workflow(tmp_path, [])
        _, results = check(tmp_path, ["components"], wf)
        suggestion_normalized = results[0]["suggestion"].replace("\\", "/")
        assert "components/cat/comp" in suggestion_normalized
        assert "          - name:" in suggestion_normalized
        assert "            context:" in suggestion_normalized

    def test_ignored_path_is_skipped(self, tmp_path):
        """Skips container files listed in the ignore file."""
        _write(tmp_path / "components/cat/comp/Containerfile", "FROM python:3.11")
        (tmp_path / IGNORE_FILENAME).write_text("components/cat/comp\n")
        wf = _make_workflow(tmp_path, [])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert all_matched
        assert results[0]["status"] == "ignored"

    def test_subcategory_path_matched(self, tmp_path):
        """Passes when a subcategory container file has a matrix entry."""
        _write(tmp_path / "components/cat/subcat/comp/Containerfile", "FROM python:3.11")
        wf = _make_workflow(tmp_path, ["components/cat/subcat/comp"])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert all_matched

    def test_missing_and_present_mixed(self, tmp_path):
        """Reports ok and unmatched correctly when results are mixed."""
        _write(tmp_path / "components/cat/a/Containerfile", "FROM a")
        _write(tmp_path / "components/cat/b/Containerfile", "FROM b")
        wf = _make_workflow(tmp_path, ["components/cat/a"])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert not all_matched
        statuses = {r["file"].replace("\\", "/").split("/")[-2]: r["status"] for r in results}
        assert statuses["a"] == "ok"
        assert statuses["b"] == "unmatched"

    def test_no_container_files_passes(self, tmp_path):
        """Passes when no container files are found."""
        wf = _make_workflow(tmp_path, [])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert all_matched
        assert results == []

    def test_duplicate_suggestion_not_emitted(self, tmp_path):
        """When both Containerfile and Dockerfile exist in same dir, suggestion shown once."""
        _write(tmp_path / "components/cat/comp/Containerfile", "FROM python:3.11")
        _write(tmp_path / "components/cat/comp/Dockerfile", "FROM python:3.11")
        wf = _make_workflow(tmp_path, [])
        all_matched, results = check(tmp_path, ["components"], wf)
        assert not all_matched
        suggestions = [r for r in results if "suggestion" in r]
        assert len(suggestions) == 1
