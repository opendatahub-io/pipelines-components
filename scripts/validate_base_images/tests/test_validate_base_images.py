"""Unit tests for validate_base_images.py."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ...lib.base_image import (
    extract_base_images,
    get_base_images_from_compile_result,
    is_valid_base_image,
    load_base_image_allowlist,
    validate_base_images,
)
from ...lib.discovery import (
    build_component_asset,
    build_pipeline_asset,
    discover_assets,
    resolve_component_path,
    resolve_pipeline_path,
)
from ...lib.kfp_compilation import compile_and_get_yaml, find_decorated_functions_runtime, load_module_from_path
from ..validate_base_images import (
    ValidationConfig,
    _collect_violations,
    _print_summary,
    _process_assets,
    get_repo_root,
    main,
    parse_args,
    process_asset,
    set_config,
)

RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture
def default_allowlist():
    """Load the default allowlist for tests that need it."""
    allowlist_path = Path(__file__).parent.parent / "base_image_allowlist.yaml"
    return load_base_image_allowlist(allowlist_path)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config before each test."""
    set_config(ValidationConfig())
    yield
    set_config(ValidationConfig())


class TestValidationConfig:
    """Tests for ValidationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.allowlist_path.name == "base_image_allowlist.yaml"


class TestIsPythonImage:
    """Tests for allowlist-driven image validation."""

    def test_python_images_allowed_by_allowlist(self, tmp_path: Path):
        """Test that Python images matching allowlist patterns are allowed."""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text(
            "\n".join(
                [
                    "allowed_images: []",
                    "allowed_image_patterns:",
                    "  - '^ghcr\\.io/kubeflow/.*$'",
                    "  - '^python:\\d+\\.\\d+.*$'",
                    "",
                ]
            )
        )
        config = ValidationConfig()
        config.allowlist_path = allowlist_file
        config.allowlist = load_base_image_allowlist(allowlist_file)

        assert is_valid_base_image("python:3.11", allowlist=config.allowlist)
        assert is_valid_base_image("python:3.11-slim", allowlist=config.allowlist)
        assert is_valid_base_image("python:3.10-alpine", allowlist=config.allowlist)
        assert is_valid_base_image("python:3.9-bullseye", allowlist=config.allowlist)

        assert not is_valid_base_image("python:latest", allowlist=config.allowlist)
        assert not is_valid_base_image("docker.io/python:3.11", allowlist=config.allowlist)
        assert is_valid_base_image("ghcr.io/kubeflow/python:3.11", allowlist=config.allowlist)
        assert not is_valid_base_image("ubuntu:22.04", allowlist=config.allowlist)

    def test_python_images_rejected_without_allowlist_entry(self, tmp_path: Path):
        """Test that Python images are rejected when not in allowlist."""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text("allowed_images: []\nallowed_image_patterns: []\n")
        config = ValidationConfig()
        config.allowlist_path = allowlist_file
        config.allowlist = load_base_image_allowlist(allowlist_file)

        assert not is_valid_base_image("python:3.11", allowlist=config.allowlist)

    def test_allowlist_invalid_regex_fails_fast(self, tmp_path: Path):
        """Test that invalid regex patterns in allowlist raise ValueError."""
        allowlist_file = tmp_path / "allowlist.yaml"
        allowlist_file.write_text(
            "\n".join(
                [
                    "allowed_images: []",
                    "allowed_image_patterns:",
                    "  - '^(python:$'",
                    "",
                ]
            )
        )
        with pytest.raises(ValueError):
            load_base_image_allowlist(allowlist_file)


class TestParseArgs:
    """Tests for argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])
        assert args.component == []
        assert args.pipeline == []

    def test_component_repeatable(self):
        """Test that --component flag can be repeated multiple times."""
        args = parse_args(["--component", "components/training/a", "--component", "components/training/b"])
        assert args.component == ["components/training/a", "components/training/b"]

    def test_pipeline_repeatable(self):
        """Test that --pipeline flag can be repeated multiple times."""
        args = parse_args(["--pipeline", "pipelines/training/a", "--pipeline", "pipelines/training/b"])
        assert args.pipeline == ["pipelines/training/a", "pipelines/training/b"]


class TestTargetResolution:
    """Tests for resolving component and pipeline paths."""

    def test_resolve_component_dir(self):
        """Test resolving a component directory path."""
        repo_root = RESOURCES_DIR
        p = resolve_component_path(repo_root, "components/training/custom_image_component")
        assert p.exists()
        assert p.name == "component.py"

    def test_resolve_pipeline_dir(self):
        """Test resolving a pipeline directory path."""
        repo_root = RESOURCES_DIR
        p = resolve_pipeline_path(repo_root, "pipelines/training/multi_image_pipeline")
        assert p.exists()
        assert p.name == "pipeline.py"

    def test_reject_path_outside_components(self):
        """Test that pipeline paths are rejected when resolving components."""
        repo_root = RESOURCES_DIR
        with pytest.raises(ValueError):
            resolve_component_path(repo_root, "pipelines/training/multi_image_pipeline")

    def test_reject_path_outside_pipelines(self):
        """Test that component paths are rejected when resolving pipelines."""
        repo_root = RESOURCES_DIR
        with pytest.raises(ValueError):
            resolve_pipeline_path(repo_root, "components/training/custom_image_component")

    def test_build_component_asset(self):
        """Test building asset metadata from a component file."""
        repo_root = RESOURCES_DIR
        component_file = resolve_component_path(repo_root, "components/training/custom_image_component")
        asset = build_component_asset(repo_root, component_file)
        assert asset["category"] == "training"
        assert asset["name"] == "custom_image_component"
        assert asset["module_path"].endswith("component.py")

    def test_build_pipeline_asset(self):
        """Test building asset metadata from a pipeline file."""
        repo_root = RESOURCES_DIR
        pipeline_file = resolve_pipeline_path(repo_root, "pipelines/training/multi_image_pipeline")
        asset = build_pipeline_asset(repo_root, pipeline_file)
        assert asset["category"] == "training"
        assert asset["name"] == "multi_image_pipeline"
        assert asset["module_path"].endswith("pipeline.py")


class TestGetRepoRoot:
    """Tests for get_repo_root function."""

    def test_returns_valid_path(self):
        """Test that get_repo_root returns an existing absolute Path."""
        result = get_repo_root()
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result.exists()


class TestDiscoverAssets:
    """Tests for discover_assets function."""

    def test_discover_components(self):
        """Test discovering components in fixture directory."""
        components_dir = RESOURCES_DIR / "components"
        assets = discover_assets(components_dir, "component")

        assert len(assets) == 7

        categories = {a["category"] for a in assets}
        assert categories == {"training", "data_processing", "validation", "edge_cases"}

        names = {a["name"] for a in assets}
        assert names == {
            "custom_image_component",
            "default_image_component",
            "valid_kubeflow_image",
            "invalid_dockerhub_image",
            "invalid_gcr_image",
            "variable_base_image",
            "functools_partial_image",
        }

    def test_discover_pipelines(self):
        """Test discovering pipelines in fixture directory."""
        pipelines_dir = RESOURCES_DIR / "pipelines"
        assets = discover_assets(pipelines_dir, "pipeline")

        assert len(assets) == 1
        assert assets[0]["category"] == "training"
        assert assets[0]["name"] == "multi_image_pipeline"

    def test_discover_nonexistent_directory(self):
        """Test discovering assets in non-existent directory."""
        assets = discover_assets(Path("/nonexistent"), "component")
        assert assets == []

    def test_discover_empty_directory(self):
        """Test discovering assets in empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            assets = discover_assets(Path(tmp_dir), "component")
            assert assets == []


class TestLoadModuleFromPath:
    """Tests for load_module_from_path function."""

    def test_load_component_module(self):
        """Test loading a component module."""
        module_path = str(RESOURCES_DIR / "components/training/custom_image_component/component.py")
        module = load_module_from_path(module_path, "test_component_module")

        assert hasattr(module, "train_model")
        assert callable(module.train_model)

    def test_load_pipeline_module(self):
        """Test loading a pipeline module."""
        module_path = str(RESOURCES_DIR / "pipelines/training/multi_image_pipeline/pipeline.py")
        module = load_module_from_path(module_path, "test_pipeline_module")

        assert hasattr(module, "training_pipeline")
        assert callable(module.training_pipeline)

    def test_load_nonexistent_module(self):
        """Test loading a non-existent module raises an exception."""
        with pytest.raises(Exception):
            load_module_from_path("/nonexistent/module.py", "nonexistent")


class TestFindDecoratedFunctions:
    """Tests for find_decorated_functions function."""

    def test_find_component_functions(self):
        """Test finding @dsl.component decorated functions."""
        module_path = str(RESOURCES_DIR / "components/training/custom_image_component/component.py")
        module = load_module_from_path(module_path, "test_find_component")

        functions = find_decorated_functions_runtime(module, "component")

        assert len(functions) == 1
        assert functions[0][0] == "train_model"

    def test_find_pipeline_functions(self):
        """Test finding @dsl.pipeline decorated functions."""
        module_path = str(RESOURCES_DIR / "pipelines/training/multi_image_pipeline/pipeline.py")
        module = load_module_from_path(module_path, "test_find_pipeline")

        functions = find_decorated_functions_runtime(module, "pipeline")

        func_names = [f[0] for f in functions]
        assert "training_pipeline" in func_names


class TestCompileAndGetYaml:
    """Tests for compile_and_get_yaml function."""

    def test_compile_component(self):
        """Test compiling a component to YAML."""
        module_path = str(RESOURCES_DIR / "components/training/custom_image_component/component.py")
        module = load_module_from_path(module_path, "test_compile_component")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/component.yaml"
            ir_yaml = compile_and_get_yaml(module.train_model, output_path)

            assert ir_yaml is not None
            assert "deploymentSpec" in ir_yaml

    def test_compile_pipeline(self):
        """Test compiling a pipeline to YAML (single- or two-doc return)."""
        module_path = str(RESOURCES_DIR / "pipelines/training/multi_image_pipeline/pipeline.py")
        module = load_module_from_path(module_path, "test_compile_pipeline")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/pipeline.yaml"
            ir_yaml = compile_and_get_yaml(module.training_pipeline, output_path)

            assert ir_yaml is not None
            # Single doc: pipeline spec at top level; two docs: wrapper with pipeline_spec key.
            if "pipeline_spec" in ir_yaml:
                spec = ir_yaml["pipeline_spec"]
                assert "deploymentSpec" in spec
                assert "root" in spec
            else:
                assert "deploymentSpec" in ir_yaml
                assert "root" in ir_yaml


class TestExtractBaseImages:
    """Tests for extract_base_images function."""

    def test_extract_custom_base_image(self):
        """Test extracting custom base image from component."""
        module_path = str(RESOURCES_DIR / "components/training/custom_image_component/component.py")
        module = load_module_from_path(module_path, "test_extract_custom")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/component.yaml"
            ir_yaml = compile_and_get_yaml(module.train_model, output_path)
            assert ir_yaml is not None
            images = get_base_images_from_compile_result(ir_yaml)

            assert "ghcr.io/kubeflow/ml-training:v1.0.0" in images

    def test_extract_default_base_image(self):
        """Test extracting default base image from component."""
        module_path = str(RESOURCES_DIR / "components/data_processing/default_image_component/component.py")
        module = load_module_from_path(module_path, "test_extract_default")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/component.yaml"
            ir_yaml = compile_and_get_yaml(module.load_data, output_path)
            assert ir_yaml is not None
            images = get_base_images_from_compile_result(ir_yaml)

            assert len(images) == 1
            assert any("python:" in img for img in images)

    def test_extract_multiple_base_images_from_pipeline(self):
        """Test extracting multiple base images from pipeline."""
        module_path = str(RESOURCES_DIR / "pipelines/training/multi_image_pipeline/pipeline.py")
        module = load_module_from_path(module_path, "test_extract_multi")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/pipeline.yaml"
            ir_yaml = compile_and_get_yaml(module.training_pipeline, output_path)
            assert ir_yaml is not None
            images = get_base_images_from_compile_result(ir_yaml)

            assert "python:3.11-slim" in images
            assert "ghcr.io/kubeflow/evaluation:v2.0.0" in images
            assert any("python:" in img for img in images)

    def test_extract_from_empty_yaml(self):
        """Test extracting from empty YAML returns empty set."""
        images = extract_base_images({})
        assert images == set()


class TestProcessAsset:
    """Tests for process_asset function."""

    def test_process_component_with_custom_image(self):
        """Test processing a component with custom base image."""
        asset = {
            "path": RESOURCES_DIR / "components/training/custom_image_component/component.py",
            "category": "training",
            "name": "custom_image_component",
            "module_path": str(RESOURCES_DIR / "components/training/custom_image_component/component.py"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = process_asset(asset, "component", tmp_dir)

            assert result["compiled"] is True
            assert not result["errors"]
            assert "ghcr.io/kubeflow/ml-training:v1.0.0" in result["base_images"]

    def test_process_component_with_default_image(self):
        """Test processing a component with default base image."""
        asset = {
            "path": RESOURCES_DIR / "components/data_processing/default_image_component/component.py",
            "category": "data_processing",
            "name": "default_image_component",
            "module_path": str(RESOURCES_DIR / "components/data_processing/default_image_component/component.py"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = process_asset(asset, "component", tmp_dir)

            assert result["compiled"] is True
            assert not result["errors"]
            assert len(result["base_images"]) == 1

    def test_process_pipeline_with_multiple_images(self):
        """Test processing a pipeline with multiple base images."""
        asset = {
            "path": RESOURCES_DIR / "pipelines/training/multi_image_pipeline/pipeline.py",
            "category": "training",
            "name": "multi_image_pipeline",
            "module_path": str(RESOURCES_DIR / "pipelines/training/multi_image_pipeline/pipeline.py"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = process_asset(asset, "pipeline", tmp_dir)

            assert result["compiled"] is True
            assert not result["errors"]
            assert "python:3.11-slim" in result["base_images"]
            assert "ghcr.io/kubeflow/evaluation:v2.0.0" in result["base_images"]

    def test_process_nonexistent_module(self):
        """Test processing a non-existent module returns error."""
        asset = {
            "path": Path("/nonexistent/component.py"),
            "category": "test",
            "name": "nonexistent",
            "module_path": "/nonexistent/component.py",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = process_asset(asset, "component", tmp_dir)

            assert result["compiled"] is False
            assert len(result["errors"]) > 0
            assert "Failed to load module" in result["errors"][0]


class TestIsValidBaseImage:
    """Tests for is_valid_base_image function."""

    def test_valid_kubeflow_image(self, default_allowlist):
        """Test that ghcr.io/kubeflow images are valid."""
        assert is_valid_base_image(
            "ghcr.io/kubeflow/pipelines-components-example:v1.0.0",
            allowlist=default_allowlist,
        )
        assert is_valid_base_image("ghcr.io/kubeflow/ml-training:latest", allowlist=default_allowlist)
        assert is_valid_base_image("ghcr.io/kubeflow/evaluation:v2.0.0", allowlist=default_allowlist)

    def test_valid_empty_image(self):
        """Test that empty/unset images are valid."""
        assert is_valid_base_image("")

    def test_valid_python_images(self, default_allowlist):
        """Test that standard Python images are valid with allowlist."""
        assert is_valid_base_image("python:3.11", allowlist=default_allowlist)
        assert is_valid_base_image("python:3.11-slim", allowlist=default_allowlist)
        assert is_valid_base_image("python:3.10", allowlist=default_allowlist)

    def test_invalid_dockerhub_image(self, default_allowlist):
        """Test that Docker Hub images are invalid."""
        assert not is_valid_base_image("docker.io/custom:latest", allowlist=default_allowlist)
        assert not is_valid_base_image("docker.io/library/python:3.11", allowlist=default_allowlist)

    def test_invalid_gcr_image(self, default_allowlist):
        """Test that GCR images are invalid."""
        assert not is_valid_base_image("gcr.io/project/image:v1.0", allowlist=default_allowlist)
        assert not is_valid_base_image("gcr.io/my-project/my-image:latest", allowlist=default_allowlist)

    def test_invalid_other_registries(self, default_allowlist):
        """Test that other registry images are invalid."""
        assert not is_valid_base_image("quay.io/some/image:tag", allowlist=default_allowlist)
        assert not is_valid_base_image("registry.example.com/image:v1", allowlist=default_allowlist)

    def test_invalid_python_variants(self, default_allowlist):
        """Test that Python images without version are invalid."""
        assert not is_valid_base_image("python:latest", allowlist=default_allowlist)
        assert not is_valid_base_image("python", allowlist=default_allowlist)

    def test_invalid_partial_kubeflow_prefix(self):
        """Test that partial kubeflow prefix is invalid."""
        assert not is_valid_base_image("ghcr.io/kubeflow")
        assert not is_valid_base_image("ghcr.io/kubeflow-fake/image:v1")


class TestValidateBaseImages:
    """Tests for validate_base_images function."""

    def test_all_valid_images(self, default_allowlist):
        """Test validation with all valid images returns empty list."""
        images = {
            "ghcr.io/kubeflow/pipelines-components-example:v1.0.0",
            "ghcr.io/kubeflow/ml-training:latest",
        }
        invalid = validate_base_images(images, allowlist=default_allowlist)
        assert invalid == set()

    def test_all_invalid_images(self, default_allowlist):
        """Test validation with all invalid images returns all."""
        images = {
            "docker.io/custom:latest",
            "gcr.io/project/image:v1.0",
        }
        invalid = validate_base_images(images, allowlist=default_allowlist)
        assert len(invalid) == 2
        assert "docker.io/custom:latest" in invalid
        assert "gcr.io/project/image:v1.0" in invalid

    def test_mixed_valid_invalid_images(self, default_allowlist):
        """Test validation with mixed images returns only invalid."""
        images = {
            "ghcr.io/kubeflow/valid:v1.0.0",
            "docker.io/custom:latest",
            "gcr.io/project/image:v1.0",
        }
        invalid = validate_base_images(images, allowlist=default_allowlist)
        assert len(invalid) == 2
        assert "ghcr.io/kubeflow/valid:v1.0.0" not in invalid
        assert "docker.io/custom:latest" in invalid
        assert "gcr.io/project/image:v1.0" in invalid

    def test_empty_set(self, default_allowlist):
        """Test validation with empty set returns empty list."""
        invalid = validate_base_images(set(), allowlist=default_allowlist)
        assert invalid == set()


class TestBaseImageValidationIntegration:
    """Integration tests for base image validation with real components."""

    def test_invalid_images_detected(self):
        """Test that invalid images (Docker Hub, GCR) are correctly detected."""
        dockerhub_asset = {
            "path": RESOURCES_DIR / "components/validation/invalid_dockerhub_image/component.py",
            "category": "validation",
            "name": "invalid_dockerhub_image",
            "module_path": str(RESOURCES_DIR / "components/validation/invalid_dockerhub_image/component.py"),
        }
        gcr_asset = {
            "path": RESOURCES_DIR / "components/validation/invalid_gcr_image/component.py",
            "category": "validation",
            "name": "invalid_gcr_image",
            "module_path": str(RESOURCES_DIR / "components/validation/invalid_gcr_image/component.py"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerhub_result = process_asset(dockerhub_asset, "component", tmp_dir)
            gcr_result = process_asset(gcr_asset, "component", tmp_dir)

            assert "docker.io/custom:latest" in dockerhub_result["invalid_base_images"]
            assert "gcr.io/project/image:v1.0" in gcr_result["invalid_base_images"]


class TestEdgeCases:
    """Tests for edge cases that demonstrate why compilation is needed."""

    def test_variable_reference_base_image(self):
        """Test component with base_image set via variable reference."""
        asset = {
            "path": RESOURCES_DIR / "components/edge_cases/variable_base_image/component.py",
            "category": "edge_cases",
            "name": "variable_base_image",
            "module_path": str(RESOURCES_DIR / "components/edge_cases/variable_base_image/component.py"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = process_asset(asset, "component", tmp_dir)

            assert result["compiled"] is True
            assert not result["errors"]
            assert "docker.io/myorg/custom-python:3.11" in result["base_images"]
            assert "docker.io/myorg/custom-python:3.11" in result["invalid_base_images"]

    def test_functools_partial_wrapper_base_image(self):
        """Test component with base_image set via functools.partial wrapper."""
        asset = {
            "path": RESOURCES_DIR / "components/edge_cases/functools_partial_image/component.py",
            "category": "edge_cases",
            "name": "functools_partial_image",
            "module_path": str(RESOURCES_DIR / "components/edge_cases/functools_partial_image/component.py"),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = process_asset(asset, "component", tmp_dir)

            assert result["compiled"] is True
            assert not result["errors"]
            assert "quay.io/myorg/python:3.11" in result["base_images"]
            assert "quay.io/myorg/python:3.11" in result["invalid_base_images"]

    def test_edge_case_images_flagged_as_violations(self, default_allowlist):
        """Test that edge case images are correctly flagged as violations."""
        assert not is_valid_base_image("docker.io/myorg/custom-python:3.11", allowlist=default_allowlist)
        assert not is_valid_base_image("quay.io/myorg/python:3.11", allowlist=default_allowlist)

    def test_edge_case_with_valid_kubeflow_image(self, default_allowlist):
        """Test that edge case patterns work with valid Kubeflow images too."""
        assert is_valid_base_image("ghcr.io/kubeflow/ml-training:v1.0.0", allowlist=default_allowlist)
        assert is_valid_base_image("ghcr.io/kubeflow/custom-runtime:latest", allowlist=default_allowlist)


class TestCollectViolations:
    """Tests for _collect_violations function."""

    def test_collect_invalid_image_violations(self):
        """Test collecting invalid image violations."""
        results = [
            {
                "path": "/path/to/comp1.py",
                "category": "training",
                "name": "comp1",
                "type": "component",
                "invalid_base_images": ["docker.io/bad1:latest", "gcr.io/bad2:v1"],
            },
            {
                "path": "/path/to/comp2.py",
                "category": "evaluation",
                "name": "comp2",
                "type": "component",
                "invalid_base_images": [],
            },
        ]
        violations = _collect_violations(results)
        assert len(violations) == 2
        assert violations[0]["image"] == "docker.io/bad1:latest"
        assert violations[1]["image"] == "gcr.io/bad2:v1"


class TestPrintSummary:
    """Tests for _print_summary function."""

    def test_print_summary_success(self, capsys):
        """Test printing summary for successful validation."""
        config = ValidationConfig()
        results = [
            {
                "compiled": True,
                "errors": [],
                "base_images": {"ghcr.io/kubeflow/valid:v1"},
                "invalid_base_images": [],
            }
        ]
        exit_code = _print_summary(results, {"ghcr.io/kubeflow/valid:v1"}, config)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "SUCCESS: All base images are valid" in captured.out

    def test_print_summary_with_violations(self, capsys):
        """Test printing summary with violations returns exit code 1."""
        config = ValidationConfig()
        results = [
            {
                "path": "/path/to/comp.py",
                "category": "training",
                "name": "comp",
                "type": "component",
                "compiled": True,
                "errors": [],
                "base_images": {"docker.io/invalid:latest"},
                "invalid_base_images": ["docker.io/invalid:latest"],
            }
        ]
        exit_code = _print_summary(results, {"docker.io/invalid:latest"}, config)
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAILED:" in captured.out

    def test_print_summary_no_assets(self, capsys):
        """Test printing summary when no assets discovered."""
        config = ValidationConfig()
        exit_code = _print_summary([], set(), config)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No components or pipelines were discovered" in captured.out
        assert "No custom base images found" not in captured.out

    def test_print_summary_failed_assets(self, capsys):
        """Test printing summary when assets fail to compile/load."""
        config = ValidationConfig()
        results = [
            {
                "path": "/path/to/comp.py",
                "category": "training",
                "name": "broken_comp",
                "type": "component",
                "compiled": False,
                "errors": ["Failed to load module: Some error"],
                "base_images": set(),
                "invalid_base_images": [],
            }
        ]
        exit_code = _print_summary(results, set(), config)
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Failed to process: 1" in captured.out
        assert "No base images could be extracted (some assets failed to compile/load)" in captured.out
        assert "FAILED: 1 asset(s) could not be processed" in captured.out

    def test_print_summary_default_images_only(self, capsys):
        """Test printing summary when assets use only default images."""
        config = ValidationConfig()
        results = [
            {
                "compiled": True,
                "errors": [],
                "base_images": set(),
                "invalid_base_images": [],
            }
        ]
        exit_code = _print_summary(results, set(), config)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Successfully compiled: 1" in captured.out
        assert "No custom base images found (all using defaults)" in captured.out
        assert "SUCCESS: All base images are valid" in captured.out


class TestProcessAssets:
    """Tests for _process_assets function."""

    def test_process_empty_assets(self):
        """Test processing empty asset list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results, images = _process_assets([], "component", "Components", tmp_dir)
            assert results == []
            assert images == set()

    def test_process_assets_with_components(self, capsys):
        """Test processing actual component assets."""
        assets = [
            {
                "path": RESOURCES_DIR / "components/training/custom_image_component/component.py",
                "category": "training",
                "name": "custom_image_component",
                "module_path": str(RESOURCES_DIR / "components/training/custom_image_component/component.py"),
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            results, images = _process_assets(assets, "component", "Components", tmp_dir)

            assert len(results) == 1
            assert results[0]["compiled"] is True
            assert "ghcr.io/kubeflow/ml-training:v1.0.0" in images

        captured = capsys.readouterr()
        assert "Processing Components" in captured.out
        assert "training/custom_image_component" in captured.out


class TestMainFunction:
    """Tests for main() function with captured output."""

    def test_main_with_resources(self, capsys):
        """Test main function running against resources directory."""
        with patch("scripts.validate_base_images.validate_base_images.get_repo_root") as mock_root:
            mock_root.return_value = RESOURCES_DIR

            exit_code = main([])

            captured = capsys.readouterr()
            assert "Kubeflow Pipelines Base Image Validator" in captured.out
            assert "Discovered" in captured.out
            assert exit_code == 1

    def test_main_with_selected_component_only(self, capsys):
        """Test main function with a specific component selected via CLI."""
        with patch("scripts.validate_base_images.validate_base_images.get_repo_root") as mock_root:
            mock_root.return_value = RESOURCES_DIR

            exit_code = main(["--component", "components/training/custom_image_component"])

            captured = capsys.readouterr()
            assert "Selected 1 component(s)" in captured.out
            assert "Selected 0 pipeline(s)" in captured.out
            assert exit_code == 0

    def test_main_empty_directory(self, capsys):
        """Test main function with empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("scripts.validate_base_images.validate_base_images.get_repo_root") as mock_root:
                mock_root.return_value = Path(tmp_dir)

                exit_code = main([])

                assert exit_code == 0
                captured = capsys.readouterr()
                assert "No components or pipelines were discovered" in captured.out


class TestCompilationFailure:
    """Tests for compilation failure handling."""

    def test_compile_invalid_function(self):
        """Test compile_and_get_yaml raises for invalid function."""

        def invalid_func():
            """Not a valid KFP component - will fail compilation."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/invalid.yaml"
            with pytest.raises(Exception):
                compile_and_get_yaml(invalid_func, output_path)
