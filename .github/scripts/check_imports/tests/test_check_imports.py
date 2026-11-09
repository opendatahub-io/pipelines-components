#!/usr/bin/env python3
"""Unit tests for check_imports.py script."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest
import yaml

from ..check_imports import (
    ImportGuardConfig,
    build_stdlib_index,
    canonicalize_module_name,
    check_imports,
    discover_python_files,
    extract_top_level_imports,
)


class TestCanonicalizeModuleName:
    """Test the canonicalize_module_name function."""

    def test_simple_module_name(self):
        """Test canonicalizing a simple module name."""
        assert canonicalize_module_name("os") == "os"
        assert canonicalize_module_name("sys") == "sys"

    def test_dotted_module_name(self):
        """Test canonicalizing a dotted module path."""
        assert canonicalize_module_name("os.path") == "os"
        assert canonicalize_module_name("collections.abc") == "collections"
        assert canonicalize_module_name("email.mime.text") == "email"

    def test_empty_string(self):
        """Test canonicalizing an empty string."""
        assert canonicalize_module_name("") == ""


class TestImportGuardConfig:
    """Test the ImportGuardConfig class."""

    def test_init_empty(self):
        """Test initializing with no allowlists."""
        config = ImportGuardConfig()
        assert config.module_allowlist == set()
        assert config.path_scoped_allowlist == {}

    def test_init_with_modules(self):
        """Test initializing with module allowlist."""
        config = ImportGuardConfig(module_allowlist=["pandas", "numpy"])
        assert "pandas" in config.module_allowlist
        assert "numpy" in config.module_allowlist

    def test_init_with_dotted_modules(self):
        """Test initializing with dotted module names."""
        config = ImportGuardConfig(module_allowlist=["pandas.core", "numpy.array"])
        assert "pandas" in config.module_allowlist
        assert "numpy" in config.module_allowlist

    def test_init_with_path_scoped(self):
        """Test initializing with path-scoped allowlist."""
        config = ImportGuardConfig(
            path_scoped_allowlist={
                "scripts": ["pytest"],
                "tests": ["mock"],
            }
        )
        scripts_path = Path("scripts").resolve()
        tests_path = Path("tests").resolve()
        assert scripts_path in config.path_scoped_allowlist
        assert tests_path in config.path_scoped_allowlist
        assert "pytest" in config.path_scoped_allowlist[scripts_path]
        assert "mock" in config.path_scoped_allowlist[tests_path]

    def test_from_path_nonexistent(self):
        """Test loading config from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ImportGuardConfig.from_path(Path("/nonexistent/file.yaml"))

    def test_from_path_valid_file(self):
        """Test loading config from valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump(
                {
                    "modules": ["pandas", "numpy"],
                    "files": {
                        "scripts": ["pytest"],
                    },
                    "exclude": ["templates/**"],
                },
                f,
            )
            config_path = Path(f.name)

        try:
            config = ImportGuardConfig.from_path(config_path)
            assert "pandas" in config.module_allowlist
            assert "numpy" in config.module_allowlist
            scripts_path = Path("scripts").resolve()
            assert scripts_path in config.path_scoped_allowlist
            assert config.excluded_paths == ("templates/**",)
        finally:
            config_path.unlink()

    def test_is_allowed_global_module(self):
        """Test checking if a module is globally allowed."""
        config = ImportGuardConfig(module_allowlist=["pandas", "numpy"])
        assert config.is_allowed("pandas", Path("any/file.py"))
        assert config.is_allowed("numpy", Path("another/file.py"))
        assert not config.is_allowed("tensorflow", Path("file.py"))

    def test_is_allowed_path_scoped(self):
        """Test checking if a module is allowed for specific path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()
            test_file = tmpdir_path / "test.py"
            test_file.touch()

            config = ImportGuardConfig(
                path_scoped_allowlist={
                    str(tmpdir_path): ["pytest"],
                }
            )

            assert config.is_allowed("pytest", test_file)
            assert not config.is_allowed("mock", test_file)

    def test_is_allowed_parent_path_scoped(self):
        """Test checking if a module is allowed via parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            test_file = subdir / "test.py"
            test_file.touch()

            config = ImportGuardConfig(
                path_scoped_allowlist={
                    str(tmpdir_path): ["pytest"],
                }
            )

            assert config.is_allowed("pytest", test_file)


class TestDiscoverPythonFiles:
    """Test the discover_python_files function."""

    def test_single_file(self):
        """Test discovering a single Python file."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            file_path = Path(f.name)

        try:
            files = discover_python_files([str(file_path)])
            assert len(files) == 1
            assert files[0] == file_path
        finally:
            file_path.unlink()

    def test_non_python_file(self):
        """Test that non-Python files are ignored."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            file_path = Path(f.name)

        try:
            files = discover_python_files([str(file_path)])
            assert len(files) == 0
        finally:
            file_path.unlink()

    def test_directory_recursive(self):
        """Test discovering Python files recursively in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "file1.py").touch()
            (tmpdir_path / "file2.py").touch()
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            (subdir / "file3.py").touch()
            (tmpdir_path / "notpython.txt").touch()

            files = discover_python_files([str(tmpdir_path)])
            assert len(files) == 3
            file_names = {f.name for f in files}
            assert file_names == {"file1.py", "file2.py", "file3.py"}

    def test_skip_hidden_files(self):
        """Test that hidden files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "visible.py").touch()
            (tmpdir_path / ".hidden.py").touch()

            files = discover_python_files([str(tmpdir_path)])
            assert len(files) == 1
            assert files[0].name == "visible.py"

    def test_skip_hidden_directories(self):
        """Test that files in hidden directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "visible.py").touch()
            hidden_dir = tmpdir_path / ".hidden"
            hidden_dir.mkdir()
            (hidden_dir / "file.py").touch()

            files = discover_python_files([str(tmpdir_path)])
            assert len(files) == 1
            assert files[0].name == "visible.py"

    def test_multiple_paths(self):
        """Test discovering files from multiple paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            dir1 = tmpdir_path / "dir1"
            dir1.mkdir()
            dir2 = tmpdir_path / "dir2"
            dir2.mkdir()

            (dir1 / "file1.py").touch()
            (dir2 / "file2.py").touch()

            files = discover_python_files([str(dir1), str(dir2)])
            assert len(files) == 2


class TestTopLevelImportVisitor:
    """Test the TopLevelImportVisitor class."""

    def test_simple_import(self):
        """Test extracting simple import statements."""
        code = "import os\nimport sys"
        tree = ast.parse(code)
        imports = extract_top_level_imports(tree)
        import_list = list(imports)
        assert len(import_list) == 2
        assert ("os", 1) in import_list
        assert ("sys", 2) in import_list

    def test_from_import(self):
        """Test extracting from...import statements."""
        code = "from os import path\nfrom collections import defaultdict"
        tree = ast.parse(code)
        imports = extract_top_level_imports(tree)
        import_list = list(imports)
        assert len(import_list) == 2
        assert ("os", 1) in import_list
        assert ("collections", 2) in import_list

    def test_dotted_import(self):
        """Test extracting dotted imports."""
        code = "import os.path\nimport email.mime.text"
        tree = ast.parse(code)
        imports = extract_top_level_imports(tree)
        import_list = list(imports)
        assert len(import_list) == 2
        assert ("os", 1) in import_list
        assert ("email", 2) in import_list

    def test_relative_import_ignored(self):
        """Test that relative imports are ignored."""
        code = "from . import module\nfrom ..package import something"
        tree = ast.parse(code)
        imports = extract_top_level_imports(tree)
        import_list = list(imports)
        assert len(import_list) == 0

    def test_function_level_import_ignored(self):
        """Test that imports inside functions are ignored."""
        code = """
import os

def foo():
    import pandas
    return pandas
"""
        tree = ast.parse(code)
        imports = extract_top_level_imports(tree)
        import_list = list(imports)
        assert len(import_list) == 1
        assert ("os", 2) in import_list

    def test_class_level_import_ignored(self):
        """Test that imports inside classes are ignored."""
        code = """
import os

class Foo:
    import numpy
"""
        tree = ast.parse(code)
        imports = extract_top_level_imports(tree)
        import_list = list(imports)
        assert len(import_list) == 1
        assert ("os", 2) in import_list


class TestBuildStdlibIndex:
    """Test the build_stdlib_index function."""

    def test_contains_common_modules(self):
        """Test that stdlib index contains common modules."""
        stdlib = build_stdlib_index()
        assert "os" in stdlib
        assert "sys" in stdlib
        assert "json" in stdlib
        assert "pathlib" in stdlib

    def test_cached(self):
        """Test that function is cached."""
        result1 = build_stdlib_index()
        result2 = build_stdlib_index()
        assert result1 is result2


class TestCheckImports:
    """Test the check_imports function."""

    def test_excluded_templates_are_not_parsed(self, tmp_path, monkeypatch, capsys):
        """Skip package-data templates without exempting adjacent KFP modules."""
        monkeypatch.chdir(tmp_path)
        template_dir = tmp_path / "templates" / "nested"
        template_dir.mkdir(parents=True)
        template = template_dir / "app.py"
        template.write_text("import fastapi\nthis is not valid python\n")
        component = tmp_path / "component.py"
        component.write_text("import fastapi\n")
        config = ImportGuardConfig(excluded_paths=["templates/**"])

        assert check_imports([template], config) == 0
        assert check_imports([component], config) == 1
        assert "imports non-stdlib module 'fastapi'" in capsys.readouterr().err

    def test_stdlib_imports_pass(self):
        """Test that stdlib imports pass validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import os\nimport sys\nfrom pathlib import Path\n")
            file_path = Path(f.name)

        try:
            config = ImportGuardConfig()
            result = check_imports([file_path], config, quiet=True)
            assert result == 0
        finally:
            file_path.unlink()

    def test_third_party_imports_fail(self):
        """Test that third-party imports fail validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import pandas\nimport numpy\n")
            file_path = Path(f.name)

        try:
            config = ImportGuardConfig()
            result = check_imports([file_path], config, quiet=True)
            assert result == 1
        finally:
            file_path.unlink()

    def test_allowed_modules_pass(self):
        """Test that allowed modules pass validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import pandas\nimport numpy\n")
            file_path = Path(f.name)

        try:
            config = ImportGuardConfig(module_allowlist=["pandas", "numpy"])
            result = check_imports([file_path], config, quiet=True)
            assert result == 0
        finally:
            file_path.unlink()

    def test_function_level_imports_pass(self):
        """Test that function-level third-party imports pass."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import os

def foo():
    import pandas
    return pandas.DataFrame()
""")
            file_path = Path(f.name)

        try:
            config = ImportGuardConfig()
            result = check_imports([file_path], config, quiet=True)
            assert result == 0
        finally:
            file_path.unlink()

    def test_syntax_error_fails(self):
        """Test that syntax errors are reported."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import os\nthis is not valid python\n")
            file_path = Path(f.name)

        try:
            config = ImportGuardConfig()
            result = check_imports([file_path], config, quiet=True)
            assert result == 1
        finally:
            file_path.unlink()

    def test_empty_file_passes(self):
        """Test that empty files pass validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            file_path = Path(f.name)

        try:
            config = ImportGuardConfig()
            result = check_imports([file_path], config, quiet=True)
            assert result == 0
        finally:
            file_path.unlink()

    def test_path_scoped_allowlist(self):
        """Test path-scoped allowlists work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()
            test_file = tmpdir_path / "test.py"
            test_file.write_text("import pytest\n")

            config = ImportGuardConfig(
                path_scoped_allowlist={
                    str(tmpdir_path): ["pytest"],
                }
            )
            result = check_imports([test_file], config, quiet=True)
            assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
