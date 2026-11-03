"""Tests for the Speculator runtime."""

import inspect

import pytest

from .. import speculator


class TestSpeculatorRuntime:
    """Tests for shared mode dispatch."""

    def test_runtime_function_exists(self):
        """Verify the runtime entry point exists."""
        assert callable(speculator.run_speculator)

    def test_runtime_supports_all_modes(self):
        """Verify all SDK modes are represented in the runtime."""
        source = inspect.getsource(speculator.run_speculator)
        for mode in ("data_only", "train_only", "offline", "online"):
            assert mode in source

    def test_runtime_forwards_configured_environment_to_trainer(self):
        """Trainer jobs receive the environment configured by the component input."""
        assert 'params["env"] = env' in inspect.getsource(speculator.run_speculator)

    def test_runtime_keeps_hugging_face_caches_off_the_kfp_workspace(self):
        """Trainer workers use writable temporary storage for Hugging Face caches."""
        source = inspect.getsource(speculator.run_speculator)
        assert 'cache_root = "/tmp/.cache/huggingface"' in source

    @pytest.mark.parametrize(
        ("configured_urls", "env", "expected"),
        [
            (None, {}, None),
            ("", {"PIP_INDEX_URL": " https://packages.example.test/simple "}, ["https://packages.example.test/simple"]),
            (
                " https://first.example.test/simple, , https://second.example.test/simple ",
                {"PIP_INDEX_URL": "https://ignored.example.test/simple"},
                ["https://first.example.test/simple", "https://second.example.test/simple"],
            ),
            ("", {"PIP_EXTRA_INDEX_URL": "https://extra.example.test/simple"}, None),
        ],
    )
    def test_pip_index_urls_can_be_configured_through_job_env(self, configured_urls, env, expected):
        """A primary PIP_INDEX_URL becomes the SDK index; extra indexes stay in env."""
        assert speculator._resolve_pip_index_urls(configured_urls, env) == expected

    @pytest.mark.parametrize(
        ("configured_urls", "env"),
        [
            ("https://user:password@packages.example.test/simple", {}),
            ("https://packages.example.test/simple?token=secret", {}),
            (
                "https://packages.example.test/simple",
                {"PIP_EXTRA_INDEX_URL": "https://user:password@extra.example.test/simple"},
            ),
        ],
    )
    def test_pip_index_urls_reject_embedded_credentials(self, configured_urls, env):
        """Reject pip index URLs that contain embedded credentials."""
        with pytest.raises(ValueError, match="must not embed credentials"):
            speculator._resolve_pip_index_urls(configured_urls, env)

    def test_runtime_publishes_checkpoint_best(self):
        """Verify Speculator publishes its best checkpoint, not an arbitrary one."""
        assert '"checkpoint_best"' in inspect.getsource(speculator.run_speculator)

    def test_runtime_validates_hidden_states_output(self):
        """Verify data-only mode validates the extracted hidden-state directory."""
        source = inspect.getsource(speculator.run_speculator)
        assert "os.path.isdir(hidden_states_dir)" in source
        assert "Check the TrainJob logs" in source

    def test_runtime_warns_without_hidden_states_artifact(self):
        """Verify PVC-only hidden-state output is explicitly reported."""
        source = inspect.getsource(speculator.run_speculator)
        assert "log.warning(" in source
        assert "no KFP output artifact is wired" in source

    def test_runtime_rejects_empty_vllm_endpoint(self):
        """Verify runtime validates endpoint configuration before selecting a runtime."""
        source = inspect.getsource(speculator.run_speculator)
        assert 'values["vllm_endpoint"] = _validate_vllm_configuration(' in source

    @pytest.mark.parametrize(
        ("vllm_source", "endpoint", "expected"),
        [
            ("managed", None, None),
            ("managed", "", None),
            ("remote", "http://vllm.example.test:8000", "http://vllm.example.test:8000"),
            ("remote", " https://vllm.example.test:443 ", "https://vllm.example.test:443"),
        ],
    )
    def test_vllm_configuration_accepts_only_matching_source_endpoint(self, vllm_source, endpoint, expected):
        """Normalize whitespace while preserving managed and remote endpoint modes."""
        assert speculator._validate_vllm_configuration(vllm_source, endpoint) == expected

    @pytest.mark.parametrize(
        ("vllm_source", "endpoint"),
        [("remote", None), ("remote", "  "), ("managed", "http://vllm.example.test:8000")],
    )
    def test_vllm_configuration_rejects_inconsistent_source_endpoint(self, vllm_source, endpoint):
        """Remote mode requires an endpoint; managed mode rejects one."""
        with pytest.raises(ValueError, match="vllm_endpoint"):
            speculator._validate_vllm_configuration(vllm_source, endpoint)

    def test_runtime_validates_raw_verifier_for_external_vllm(self):
        """Verify external vLLM validates PVC-backed paths after URI normalization."""
        source = inspect.getsource(speculator.run_speculator)
        assert 'raw_verifier_model = values["verifier_model"]' in source
        assert "_validate_external_pvc_paths(" in source

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("models/Qwen3-0.6B", "pvc://shared/models/Qwen3-0.6B"),
            ("pvc://shared/models/Qwen3-0.6B", "pvc://shared/models/Qwen3-0.6B"),
        ],
    )
    def test_trainer_path_accepts_relative_and_pvc_uri_forms(self, value, expected):
        """Bare and explicit PVC paths resolve to the same Trainer URI."""
        assert speculator._trainer_path(value, "shared", "workspace") == expected

    @pytest.mark.parametrize("value", ["speculator-output", "pvc://shared/speculator-output"])
    def test_local_path_accepts_relative_and_pvc_uri_forms(self, tmp_path, value):
        """Bare and explicit PVC output paths resolve to the same shared mount."""
        persistent_mount = str(tmp_path / "persistent")
        workspace_mount = str(tmp_path / "workspace")
        assert speculator._local_path(value, "shared", persistent_mount, "workspace", workspace_mount) == str(
            tmp_path / "persistent" / "speculator-output"
        )

    @pytest.mark.parametrize("value", ["../outside", "pvc://shared/../outside"])
    def test_storage_paths_cannot_escape_mounted_pvc_roots(self, tmp_path, value):
        """Bare paths and Trainer PVC URIs reject parent-directory traversal."""
        with pytest.raises(ValueError, match="relative PVC path"):
            speculator._trainer_path(value, "shared", "workspace")
        with pytest.raises(ValueError, match="relative PVC path"):
            speculator._local_path(value, "shared", str(tmp_path / "persistent"), "workspace", str(tmp_path))

    @pytest.mark.parametrize(
        ("raw_verifier", "verifier_model_pvc"),
        [
            ("models/verifier", "models/verifier"),
            ("pvc://shared/models/verifier", "pvc://shared/models/verifier"),
        ],
    )
    def test_external_vllm_accepts_bare_and_explicit_pvc_verifier_paths(self, raw_verifier, verifier_model_pvc):
        """External vLLM accepts either path form after Trainer URI resolution."""
        trainer_verifier = speculator._trainer_path(raw_verifier, "shared", "workspace")
        trainer_hidden_states = speculator._trainer_path("speculator-output/hidden_states", "shared", "workspace")

        speculator._validate_external_pvc_paths(
            raw_verifier,
            verifier_model_pvc,
            trainer_verifier,
            trainer_hidden_states,
        )

    def test_external_vllm_rejects_hugging_face_verifier_without_pvc_path(self):
        """External vLLM still requires a verifier path accessible to the remote server."""
        with pytest.raises(ValueError, match="PVC-backed"):
            speculator._validate_external_pvc_paths(
                "Qwen/Qwen2.5-1.5B-Instruct",
                None,
                "pvc://shared/Qwen/Qwen2.5-1.5B-Instruct",
                "pvc://shared/speculator-output/hidden_states",
            )
