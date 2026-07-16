"""Tests for refresh_pipeline_requirements script."""

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from ..refresh_pipeline_requirements import (
    RefreshRequirementsError,
    build_container_command,
    build_pip_platform_args,
    build_volume_mount,
    compile_pipeline_requirements,
    read_index_url,
    refresh_pipeline_requirements,
    resolve_container_runtime,
    resolve_pipeline_dir,
    sanitize_index_url_for_log,
)


def _write_requirements_in(directory: Path, index_url: str = "https://example.com/simple") -> None:
    directory.mkdir(parents=True)
    (directory / "requirements.in").write_text(
        f"--index-url {index_url}\n\nrequests\n",
        encoding="utf-8",
    )


class TestReadIndexUrl:
    """Tests for read_index_url."""

    def test_reads_index_url(self, tmp_path: Path):
        """Returns the index URL when requirements.in declares one."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text(
            "--index-url https://example.com/simple\n\nrequests\n",
            encoding="utf-8",
        )

        assert read_index_url(requirements_in) == "https://example.com/simple"

    def test_returns_none_when_missing(self, tmp_path: Path):
        """Returns None when requirements.in has no index URL."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text("requests\n", encoding="utf-8")

        assert read_index_url(requirements_in) is None


class TestSanitizeIndexUrlForLog:
    """Tests for sanitize_index_url_for_log."""

    def test_strips_path_and_query_without_credentials(self):
        """Removes path and query from credential-free URLs."""
        url = "https://console.redhat.com/api/pypi/public-rhai/rhoai/3.5/cpu-ubi9-test/simple"

        assert sanitize_index_url_for_log(url) == "https://console.redhat.com"

    def test_strips_embedded_userinfo(self):
        """Removes username and password from the logged URL."""
        url = "https://user:secret@example.com/simple"

        assert sanitize_index_url_for_log(url) == "https://example.com"

    def test_strips_userinfo_and_preserves_port(self):
        """Removes credentials while preserving an explicit port."""
        url = "https://user:secret@example.com:8443/simple?foo=bar"

        assert sanitize_index_url_for_log(url) == "https://example.com:8443"


class TestBuildPipPlatformArgs:
    """Tests for build_pip_platform_args."""

    def test_converts_uv_platform_to_pip_args(self):
        """Converts uv-style platform strings to pip resolver arguments."""
        args = build_pip_platform_args("x86_64-manylinux_2_34", "3.12")

        assert args == ("--platform manylinux_2_34_x86_64 --python-version 3.12 --implementation cp --abi cp312")

    def test_rejects_invalid_platform(self):
        """Raises when the platform string is malformed."""
        with pytest.raises(RefreshRequirementsError, match="Invalid python platform"):
            build_pip_platform_args("manylinux_2_34", "3.12")

    def test_rejects_invalid_python_version(self):
        """Raises when the python version string is malformed."""
        with pytest.raises(RefreshRequirementsError, match="Invalid python version"):
            build_pip_platform_args("x86_64-manylinux_2_34", "3")


class TestResolvePipelineDir:
    """Tests for resolve_pipeline_dir."""

    def test_resolves_relative_pipeline_path(self, tmp_path: Path):
        """Resolves a pipeline path relative to the repository root."""
        pipeline_dir = tmp_path / "pipelines" / "training" / "demo"
        _write_requirements_in(pipeline_dir)

        resolved = resolve_pipeline_dir(tmp_path, "pipelines/training/demo")

        assert resolved == pipeline_dir.resolve()

    def test_raises_when_requirements_in_missing(self, tmp_path: Path):
        """Raises when the pipeline directory lacks requirements.in."""
        pipeline_dir = tmp_path / "pipelines" / "training" / "demo"
        pipeline_dir.mkdir(parents=True)

        with pytest.raises(RefreshRequirementsError, match="Missing requirements.in"):
            resolve_pipeline_dir(tmp_path, pipeline_dir)


class TestResolveContainerRuntime:
    """Tests for resolve_container_runtime."""

    def test_uses_explicit_runtime(self):
        """Returns the runtime requested on the command line."""
        with mock.patch("shutil.which", return_value="/usr/bin/docker"):
            assert resolve_container_runtime("docker") == "docker"

    def test_uses_environment_variable(self):
        """Returns the runtime from CONTAINER_RUNTIME when set."""
        with (
            mock.patch.dict("os.environ", {"CONTAINER_RUNTIME": "docker"}),
            mock.patch("shutil.which", return_value="/usr/bin/docker"),
        ):
            assert resolve_container_runtime() == "docker"

    def test_prefers_podman_when_auto_detecting(self):
        """Auto-detects podman before docker when both are available."""
        with mock.patch(
            "shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}" if name in {"podman", "docker"} else None,
        ):
            assert resolve_container_runtime() == "podman"

    def test_falls_back_to_docker_when_podman_missing(self):
        """Auto-detects docker when podman is unavailable."""
        with mock.patch("shutil.which", side_effect=lambda name: "/usr/bin/docker" if name == "docker" else None):
            assert resolve_container_runtime() == "docker"

    def test_raises_when_runtime_missing_on_path(self):
        """Raises when an explicit runtime is not installed."""
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RefreshRequirementsError, match="not found on PATH"):
                resolve_container_runtime("docker")

    def test_raises_when_no_runtime_available(self):
        """Raises when auto-detection finds no supported runtime."""
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RefreshRequirementsError, match="No supported container runtime found"):
                resolve_container_runtime()


class TestBuildVolumeMount:
    """Tests for build_volume_mount."""

    def test_builds_bind_mount(self, tmp_path: Path):
        """Builds a bind mount for the pipeline directory."""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()

        mount = build_volume_mount(pipeline_dir.resolve())

        assert mount == f"{pipeline_dir.resolve()}:{pipeline_dir.resolve()}"


class TestBuildContainerCommand:
    """Tests for build_container_command."""

    def test_builds_expected_podman_command(self, tmp_path: Path):
        """Builds a verbose Podman command with Hermeto-compatible pip-compile flags."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_container_command(
            runtime="podman",
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=True,
            dry_run=False,
            verbose=True,
        )

        assert command[0] == "podman"
        assert "/bin/bash" in command
        assert "-lc" in command
        compile_command = command[command.index("-lc") + 1]
        assert "python3 -u -m piptools compile requirements.in" in compile_command
        assert "python3 -u -m pip install --index-url https://pypi.org/simple pip-tools" in compile_command
        assert "--generate-hashes" in compile_command
        assert "--allow-unsafe" in compile_command
        assert "--no-header" in compile_command
        assert " -v" in compile_command
        assert "--upgrade" in compile_command
        assert "--pip-args" in compile_command
        assert "--platform manylinux_2_34_x86_64" in compile_command
        assert "--python-version 3.12" in compile_command
        assert "--abi cp312" in compile_command
        assert "--output-file requirements.txt" in compile_command
        assert "--dry-run" not in compile_command
        assert "--quiet" not in compile_command

    def test_no_upgrade_omits_upgrade_flag(self, tmp_path: Path):
        """Omits --upgrade when upgrade is disabled."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_container_command(
            runtime="podman",
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=False,
            dry_run=False,
            verbose=True,
        )

        compile_command = command[command.index("-lc") + 1]
        assert "--upgrade" not in compile_command

    def test_builds_expected_docker_command(self, tmp_path: Path):
        """Builds a Docker command with a plain bind mount."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_container_command(
            runtime="docker",
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=False,
            dry_run=False,
            verbose=False,
        )

        volume_mount = command[command.index("-v") + 1]
        assert command[0] == "docker"
        assert volume_mount == f"{pipeline_dir.resolve()}:{pipeline_dir.resolve()}"

    def test_quiet_mode_omits_verbose_flags(self, tmp_path: Path):
        """Omits verbose flags when quiet mode is requested."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_container_command(
            runtime="podman",
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=False,
            dry_run=False,
            verbose=False,
        )

        compile_command = command[command.index("-lc") + 1]
        assert " -v" not in compile_command
        assert "--quiet" in compile_command
        assert "--index-url https://pypi.org/simple" in compile_command
        assert "python3 -u -m piptools compile requirements.in" in compile_command

    def test_includes_upgrade_and_dry_run_flags(self, tmp_path: Path):
        """Passes upgrade and dry-run flags through to pip-compile."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_container_command(
            runtime="podman",
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=True,
            dry_run=True,
            verbose=False,
        )

        assert "--upgrade" in command[-1]
        assert "--dry-run" in command[-1]


class TestCompilePipelineRequirements:
    """Tests for compile_pipeline_requirements."""

    def test_quiet_mode_suppresses_container_output(self, tmp_path: Path):
        """Discards container stdout/stderr when quiet mode is requested."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run") as run_mock,
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", verbose=False)

        run_kwargs = run_mock.call_args.kwargs
        assert run_kwargs["stdout"] is subprocess.DEVNULL
        assert run_kwargs["stderr"] is subprocess.DEVNULL

    def test_runs_container_compile(self, tmp_path: Path):
        """Invokes the container runtime to run pip-compile for a valid pipeline directory."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir, index_url="https://user:secret@example.com/simple")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run") as run_mock,
            mock.patch("builtins.print") as print_mock,
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker")

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        assert command[0] == "docker"
        assert "python3 -u -m piptools compile requirements.in" in command[-1]
        printed = " ".join(str(call.args[0]) for call in print_mock.call_args_list)
        assert "user:secret" not in printed
        assert "https://example.com" in printed

    def test_requires_index_url(self, tmp_path: Path):
        """Raises when requirements.in does not declare an index URL."""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        (pipeline_dir / "requirements.in").write_text("requests\n", encoding="utf-8")

        with pytest.raises(RefreshRequirementsError, match="must declare --index-url"):
            compile_pipeline_requirements(pipeline_dir, container_runtime="podman")


class TestRefreshPipelineRequirements:
    """Tests for refresh_pipeline_requirements."""

    def test_refreshes_all_default_pipelines(self, tmp_path: Path):
        """Compiles requirements for each requested pipeline directory."""
        pipelines = [
            tmp_path / "pipelines/training/automl/demo",
            tmp_path / "pipelines/training/autorag/demo",
        ]
        for pipeline_dir in pipelines:
            _write_requirements_in(pipeline_dir)

        with mock.patch(
            "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.compile_pipeline_requirements"
        ) as compile_mock:
            refresh_pipeline_requirements(
                [str(p.relative_to(tmp_path)) for p in pipelines],
                repo_root=tmp_path,
            )

        assert compile_mock.call_count == 2
