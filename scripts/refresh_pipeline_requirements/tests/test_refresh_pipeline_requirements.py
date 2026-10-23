"""Tests for refresh_pipeline_requirements script."""

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from ..refresh_pipeline_requirements import (
    RefreshRequirementsError,
    apply_environment_markers,
    build_container_command,
    build_volume_mount,
    compile_pipeline_requirements,
    read_index_url,
    read_requirement_markers,
    refresh_pipeline_requirements,
    resolve_container_runtime,
    resolve_pipeline_dir,
    sanitize_index_url_for_log,
    strip_lockfile_hashes,
    verify_pipeline_lockfile,
    verify_pipeline_requirements,
)
from ..verify_lockfile import Finding


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


class TestReadRequirementMarkers:
    """Tests for read_requirement_markers."""

    def test_reads_platform_marker(self, tmp_path: Path):
        """Returns the environment marker declared on a pinned package."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text(
            "--index-url https://example.com/simple\n\ntriton==3.6.0; platform_machine != 's390x'\n",
            encoding="utf-8",
        )

        assert read_requirement_markers(requirements_in) == {"triton": "platform_machine != 's390x'"}

    def test_ignores_unmarked_packages_and_comments(self, tmp_path: Path):
        """Skips option lines, comments, and packages without markers."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text(
            "--index-url https://example.com/simple\n"
            "kfp==2.16.1 # from kfp-components\n"
            "triton==3.6.0; platform_machine != 's390x'\n"
            "# ignored\n",
            encoding="utf-8",
        )

        assert read_requirement_markers(requirements_in) == {"triton": "platform_machine != 's390x'"}

    def test_canonicalizes_package_name(self, tmp_path: Path):
        """Normalizes dotted/underscored names to PEP 503 form."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text(
            "Foo_Bar==1.0; python_version >= '3.11'\n",
            encoding="utf-8",
        )

        assert read_requirement_markers(requirements_in) == {"foo-bar": "python_version >= '3.11'"}


class TestApplyEnvironmentMarkers:
    """Tests for apply_environment_markers."""

    def test_injects_marker_on_hashed_pin(self, tmp_path: Path):
        """Adds the input marker to the compiled pin without changing hashes."""
        requirements_in = tmp_path / "requirements.in"
        requirements_txt = tmp_path / "requirements.txt"
        requirements_in.write_text(
            "triton==3.6.0; platform_machine != 's390x'\n",
            encoding="utf-8",
        )
        requirements_txt.write_text(
            "triton==3.6.0 \\\n    --hash=sha256:abc\n    # via -r requirements.in\n",
            encoding="utf-8",
        )

        apply_environment_markers(requirements_in, requirements_txt)

        assert requirements_txt.read_text(encoding="utf-8") == (
            "triton==3.6.0 ; platform_machine != 's390x' \\\n    --hash=sha256:abc\n    # via -r requirements.in\n"
        )

    def test_is_idempotent(self, tmp_path: Path):
        """Does not duplicate a marker that is already present."""
        requirements_in = tmp_path / "requirements.in"
        requirements_txt = tmp_path / "requirements.txt"
        requirements_in.write_text(
            "triton==3.6.0; platform_machine != 's390x'\n",
            encoding="utf-8",
        )
        requirements_txt.write_text(
            "triton==3.6.0 ; platform_machine != 's390x' \\\n    --hash=sha256:abc\n",
            encoding="utf-8",
        )

        apply_environment_markers(requirements_in, requirements_txt)
        apply_environment_markers(requirements_in, requirements_txt)

        assert requirements_txt.read_text(encoding="utf-8") == (
            "triton==3.6.0 ; platform_machine != 's390x' \\\n    --hash=sha256:abc\n"
        )

    def test_matches_canonical_lockfile_name(self, tmp_path: Path):
        """Applies markers when the lockfile uses the PEP 503 package name."""
        requirements_in = tmp_path / "requirements.in"
        requirements_txt = tmp_path / "requirements.txt"
        requirements_in.write_text(
            "autogluon.tabular==1.5.0; python_version >= '3.11'\n",
            encoding="utf-8",
        )
        requirements_txt.write_text(
            "autogluon-tabular[fastai]==1.5.0 \\\n    --hash=sha256:abc\n",
            encoding="utf-8",
        )

        apply_environment_markers(requirements_in, requirements_txt)

        assert "autogluon-tabular[fastai]==1.5.0 ; python_version >= '3.11' \\" in requirements_txt.read_text(
            encoding="utf-8"
        )


class TestSanitizeIndexUrlForLog:
    """Tests for sanitize_index_url_for_log."""

    def test_strips_path_and_query_without_credentials(self):
        """Removes path and query from credential-free URLs."""
        url = "https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6/cpu-ubi9-test/simple"

        assert sanitize_index_url_for_log(url) == "https://console.redhat.com"

    def test_strips_embedded_userinfo(self):
        """Removes username and password from the logged URL."""
        url = "https://user:secret@example.com/simple"

        assert sanitize_index_url_for_log(url) == "https://example.com"

    def test_strips_userinfo_and_preserves_port(self):
        """Removes credentials while preserving an explicit port."""
        url = "https://user:secret@example.com:8443/simple?foo=bar"

        assert sanitize_index_url_for_log(url) == "https://example.com:8443"


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
        """Builds a verbose Podman command with Hermeto-compatible uv pip compile flags."""
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
        assert "python3 -u -m uv pip compile requirements.in" in compile_command
        assert "python3 -u -m pip install uv" in compile_command
        assert "--generate-hashes" in compile_command
        assert "--emit-index-url" in compile_command
        assert "--no-header" in compile_command
        assert "--upgrade" in compile_command
        assert "--output-file requirements.txt" in compile_command
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
        assert "--quiet" in compile_command
        assert "python3 -u -m uv pip compile requirements.in" in compile_command

    def test_dry_run_omits_output_file(self, tmp_path: Path):
        """Emits to stdout (no --output-file) in dry-run mode, since uv has no --dry-run."""
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
        assert "--output-file" not in command[-1]


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
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", verbose=False, verify=False)

        run_kwargs = run_mock.call_args.kwargs
        assert run_kwargs["stdout"] is subprocess.DEVNULL
        assert run_kwargs["stderr"] is subprocess.DEVNULL

    def test_runs_container_compile(self, tmp_path: Path):
        """Invokes the container runtime to run uv pip compile for a valid pipeline directory."""
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
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", verify=False)

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        assert command[0] == "docker"
        assert "python3 -u -m uv pip compile requirements.in" in command[-1]
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

    def test_restores_environment_markers_after_compile(self, tmp_path: Path):
        """Re-applies requirements.in markers that uv pip compile strips."""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        (pipeline_dir / "requirements.in").write_text(
            "--index-url https://example.com/simple\n\ntriton==3.6.0; platform_machine != 's390x'\n",
            encoding="utf-8",
        )

        def fake_run(*_args, **_kwargs):
            (pipeline_dir / "requirements.txt").write_text(
                "triton==3.6.0 \\\n    --hash=sha256:abc\n",
                encoding="utf-8",
            )
            return mock.MagicMock(returncode=0)

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run", side_effect=fake_run),
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", verify=False)

        assert (pipeline_dir / "requirements.txt").read_text(encoding="utf-8") == (
            "triton==3.6.0 ; platform_machine != 's390x' \\\n    --hash=sha256:abc\n"
        )

    def test_dry_run_does_not_rewrite_lockfile(self, tmp_path: Path):
        """Leaves requirements.txt unchanged in dry-run mode."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        lockfile = pipeline_dir / "requirements.txt"
        lockfile.write_text("requests==2.0.0\n", encoding="utf-8")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run") as run_mock,
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", dry_run=True)

        run_mock.assert_called_once()
        assert lockfile.read_text(encoding="utf-8") == "requests==2.0.0\n"

    def test_verifies_lockfile_after_compile(self, tmp_path: Path):
        """Verifies the compiled lockfile against the index after markers are applied."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        def fake_run(*_args, **_kwargs):
            (pipeline_dir / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")
            return mock.MagicMock(returncode=0)

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run", side_effect=fake_run),
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_pipeline_lockfile"
            ) as verify_mock,
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", arch_check="warn")

        verify_mock.assert_called_once_with(pipeline_dir, arch_check="warn")

    def test_no_verify_skips_verification(self, tmp_path: Path):
        """Skips lockfile verification when verify is disabled."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        def fake_run(*_args, **_kwargs):
            (pipeline_dir / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")
            return mock.MagicMock(returncode=0)

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run", side_effect=fake_run),
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_pipeline_lockfile"
            ) as verify_mock,
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", verify=False)

        verify_mock.assert_not_called()

    def test_dry_run_skips_verification(self, tmp_path: Path):
        """Skips lockfile verification in dry-run mode since no lockfile is written."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run"),
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_pipeline_lockfile"
            ) as verify_mock,
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", dry_run=True)

        verify_mock.assert_not_called()

    def test_no_upgrade_strips_hashes_before_compile(self, tmp_path: Path):
        """Strips existing hashes before a no-upgrade compile so uv regenerates them."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        lockfile = pipeline_dir / "requirements.txt"
        lockfile.write_text("requests==2.0.0 \\\n    --hash=sha256:abc\n", encoding="utf-8")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run"),
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", upgrade=False, verify=False)

        assert lockfile.read_text(encoding="utf-8") == "requests==2.0.0\n"

    def test_failed_compile_restores_original_lockfile(self, tmp_path: Path):
        """Restores the pre-strip lockfile when the compile command fails."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        lockfile = pipeline_dir / "requirements.txt"
        original = "requests==2.0.0 \\\n    --hash=sha256:abc\n"
        lockfile.write_text(original, encoding="utf-8")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(returncode=2, cmd=["uv"])),
            mock.patch("builtins.print"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                compile_pipeline_requirements(pipeline_dir, container_runtime="docker", upgrade=False, verify=False)

        assert lockfile.read_text(encoding="utf-8") == original

    def test_missing_runtime_leaves_lockfile_untouched(self, tmp_path: Path):
        """Never strips hashes when container runtime resolution fails first."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        lockfile = pipeline_dir / "requirements.txt"
        original = "requests==2.0.0 \\\n    --hash=sha256:abc\n"
        lockfile.write_text(original, encoding="utf-8")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                side_effect=RuntimeError("no container runtime found"),
            ),
            mock.patch("subprocess.run") as run_mock,
            mock.patch("builtins.print"),
        ):
            with pytest.raises(RuntimeError, match="no container runtime found"):
                compile_pipeline_requirements(pipeline_dir, container_runtime=None, upgrade=False, verify=False)

        run_mock.assert_not_called()
        assert lockfile.read_text(encoding="utf-8") == original

    def test_upgrade_keeps_existing_hashes(self, tmp_path: Path):
        """Leaves the existing lockfile untouched before an upgrade compile."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        lockfile = pipeline_dir / "requirements.txt"
        original = "requests==2.0.0 \\\n    --hash=sha256:abc\n"
        lockfile.write_text(original, encoding="utf-8")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run"),
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", verify=False)

        assert lockfile.read_text(encoding="utf-8") == original

    def test_dry_run_does_not_strip_hashes(self, tmp_path: Path):
        """Leaves the lockfile untouched in dry-run mode even without an upgrade."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        lockfile = pipeline_dir / "requirements.txt"
        original = "requests==2.0.0 \\\n    --hash=sha256:abc\n"
        lockfile.write_text(original, encoding="utf-8")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.resolve_container_runtime",
                return_value="docker",
            ),
            mock.patch("subprocess.run"),
            mock.patch("builtins.print"),
        ):
            compile_pipeline_requirements(pipeline_dir, container_runtime="docker", upgrade=False, dry_run=True)

        assert lockfile.read_text(encoding="utf-8") == original


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


class TestStripLockfileHashes:
    """Tests for strip_lockfile_hashes."""

    def test_strips_hashes_and_dangling_continuations(self, tmp_path: Path):
        r"""Removes --hash lines and the "\" continuations that pointed at them."""
        requirements_txt = tmp_path / "requirements.txt"
        requirements_txt.write_text(
            "--index-url https://example.com/simple\n"
            "\n"
            "pillow==13.0.0 \\\n"
            "    --hash=sha256:1111 \\\n"
            "    --hash=sha256:2222\n"
            "    # via\n"
            "    #   fsspec\n"
            'triton==3.6.0 ; platform_machine != "s390x" \\\n'
            "    --hash=sha256:3333\n"
            "aiohttp==3.14.5 \\\n"
            "    --hash=sha256:4444\n",
            encoding="utf-8",
        )

        strip_lockfile_hashes(requirements_txt)

        assert requirements_txt.read_text(encoding="utf-8") == (
            "--index-url https://example.com/simple\n"
            "\n"
            "pillow==13.0.0\n"
            "    # via\n"
            "    #   fsspec\n"
            'triton==3.6.0 ; platform_machine != "s390x"\n'
            "aiohttp==3.14.5\n"
        )

    def test_is_idempotent(self, tmp_path: Path):
        """Leaves an already-stripped lockfile unchanged."""
        requirements_txt = tmp_path / "requirements.txt"
        stripped = "--index-url https://example.com/simple\n\npillow==13.0.0\n    # via fsspec\n"
        requirements_txt.write_text(stripped, encoding="utf-8")

        strip_lockfile_hashes(requirements_txt)
        strip_lockfile_hashes(requirements_txt)

        assert requirements_txt.read_text(encoding="utf-8") == stripped


class TestVerifyPipelineLockfile:
    """Tests for verify_pipeline_lockfile."""

    def _write_pipeline(self, tmp_path: Path) -> Path:
        """Create a pipeline directory with requirements.in and requirements.txt."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)
        (pipeline_dir / "requirements.txt").write_text("pillow==13.0.0\n", encoding="utf-8")
        return pipeline_dir

    def test_raises_on_error_findings(self, tmp_path: Path):
        """Raises RefreshRequirementsError when verification reports errors."""
        pipeline_dir = self._write_pipeline(tmp_path)
        findings = [
            Finding("error", "pillow==13.0.0", "missing --hash for pillow-13.0.0-1-cp312-cp312-linux_ppc64le.whl")
        ]

        with (
            mock.patch("scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.SimpleIndexClient"),
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_lockfile_coverage",
                return_value=findings,
            ),
            mock.patch("builtins.print"),
        ):
            with pytest.raises(RefreshRequirementsError, match="verification failed with 1 error"):
                verify_pipeline_lockfile(pipeline_dir)

    def test_warns_but_passes_on_warning_findings(self, tmp_path: Path):
        """Prints warnings but does not raise when findings are warnings only."""
        pipeline_dir = self._write_pipeline(tmp_path)
        findings = [Finding("warning", "xgboost==3.3.0", "index publishes wheels only for x86_64, aarch64")]

        with (
            mock.patch("scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.SimpleIndexClient"),
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_lockfile_coverage",
                return_value=findings,
            ) as coverage_mock,
            mock.patch("builtins.print") as print_mock,
        ):
            verify_pipeline_lockfile(pipeline_dir, arch_check="warn")

        assert coverage_mock.call_args.kwargs["arch_check"] == "warn"
        printed = [str(call.args[0]) for call in print_mock.call_args_list]
        assert any("warning: xgboost==3.3.0" in line for line in printed)

    def test_requires_index_url(self, tmp_path: Path):
        """Raises when requirements.in declares no index URL."""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        (pipeline_dir / "requirements.in").write_text("requests\n", encoding="utf-8")
        (pipeline_dir / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")

        with pytest.raises(RefreshRequirementsError, match="must declare --index-url"):
            verify_pipeline_lockfile(pipeline_dir)

    def test_requires_lockfile(self, tmp_path: Path):
        """Raises when the pipeline directory has no requirements.txt."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        with pytest.raises(RefreshRequirementsError, match="Missing requirements.txt"):
            verify_pipeline_lockfile(pipeline_dir)


class TestVerifyPipelineRequirements:
    """Tests for verify_pipeline_requirements."""

    def test_verifies_each_pipeline(self, tmp_path: Path):
        """Verifies the lockfile of every requested pipeline directory."""
        pipelines = [
            tmp_path / "pipelines/training/automl/demo",
            tmp_path / "pipelines/training/autorag/demo",
        ]
        for pipeline_dir in pipelines:
            _write_requirements_in(pipeline_dir)

        with mock.patch(
            "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_pipeline_lockfile"
        ) as verify_mock:
            verify_pipeline_requirements(
                [str(p.relative_to(tmp_path)) for p in pipelines],
                repo_root=tmp_path,
                arch_check="warn",
            )

        assert verify_mock.call_count == 2
        assert all(call.kwargs == {"arch_check": "warn"} for call in verify_mock.call_args_list)

    def test_verifies_remaining_pipelines_after_failure(self, tmp_path: Path):
        """Reports failures from every pipeline instead of stopping at the first one."""
        pipelines = [
            tmp_path / "pipelines/training/automl/demo",
            tmp_path / "pipelines/training/autorag/demo",
        ]
        for pipeline_dir in pipelines:
            _write_requirements_in(pipeline_dir)

        def failing_verify(pipeline_dir, **_kwargs):
            raise RefreshRequirementsError(f"verification failed for {pipeline_dir.name}")

        with (
            mock.patch(
                "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.verify_pipeline_lockfile",
                side_effect=failing_verify,
            ),
            mock.patch("builtins.print"),
        ):
            with pytest.raises(RefreshRequirementsError, match="failed for 2 of 2"):
                verify_pipeline_requirements([str(p) for p in pipelines], repo_root=tmp_path)
