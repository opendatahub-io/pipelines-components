"""Tests for lockfile verification against the simple index."""

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from ..verify_lockfile import (
    ARCH_LESS,
    LINUX_ARCHES,
    OTHER_PLATFORM,
    SOURCE,
    IndexFile,
    IndexLookupError,
    SimpleIndexClient,
    allowed_arches,
    file_arch,
    marker_allows_arch,
    parse_lockfile_pins,
    verify_lockfile_coverage,
)


def _index_file(filename: str, sha256: str | None = None) -> IndexFile:
    """Build an IndexFile with a unique deterministic digest derived from the filename."""
    digest = sha256 or hashlib.sha256(filename.encode()).hexdigest()
    return IndexFile(filename=filename, sha256=digest, arch=file_arch(filename))


class FakeIndexClient:
    """In-memory stand-in for SimpleIndexClient keyed by canonical package name."""

    def __init__(self, files: dict[str, list[IndexFile]] | None = None, error: str | None = None) -> None:
        """Configure the files served per package or the error to raise."""
        self.files_by_package = files or {}
        self.error = error

    def files(self, package: str) -> list[IndexFile]:
        """Return the configured files for a package or raise the configured error."""
        if self.error is not None:
            raise IndexLookupError(self.error)
        return self.files_by_package.get(package, [])


def _write_lockfile(tmp_path: Path, content: str) -> Path:
    """Write lockfile content to a temporary requirements.txt and return its path."""
    requirements_txt = tmp_path / "requirements.txt"
    requirements_txt.write_text(content, encoding="utf-8")
    return requirements_txt


def _pillow_files(with_ppc64le: bool = True) -> list[IndexFile]:
    """Index files for pillow==13.0.0 covering the four Linux architectures."""
    files = [
        _index_file("pillow-13.0.0-1-cp312-cp312-linux_x86_64.whl"),
        _index_file("pillow-13.0.0-1-cp312-cp312-linux_aarch64.whl"),
        _index_file("pillow-13.0.0-1-cp312-cp312-linux_s390x.whl"),
    ]
    if with_ppc64le:
        files.append(_index_file("pillow-13.0.0-1-cp312-cp312-linux_ppc64le.whl"))
    return files


def _pillow_hashes(files: list[IndexFile], skip: IndexFile | None = None) -> str:
    """Render --hash lines for the given index files, optionally skipping one."""
    lines = []
    for index_file in files:
        if index_file is skip:
            continue
        lines.append(f"    --hash=sha256:{index_file.sha256} \\")
    if lines:
        lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


class TestParseLockfilePins:
    """Tests for parse_lockfile_pins."""

    def test_parses_pins_markers_and_hashes(self, tmp_path: Path):
        """Extracts names, versions, markers, and hashes from a compiled lockfile."""
        requirements_txt = _write_lockfile(
            tmp_path,
            "--index-url https://example.com/simple\n"
            "\n"
            "pillow==13.0.0 \\\n"
            "    --hash=sha256:1111 \\\n"
            "    --hash=sha256:2222\n"
            "    # via\n"
            "    #   fsspec\n"
            'triton==3.6.0 ; platform_machine != "s390x" \\\n'
            "    --hash=sha256:3333\n",
        )

        pins = parse_lockfile_pins(requirements_txt)

        assert [(pin.name, pin.version) for pin in pins] == [("pillow", "13.0.0"), ("triton", "3.6.0")]
        assert pins[0].display == "pillow==13.0.0"
        assert pins[0].marker is None
        assert pins[0].hashes == {"sha256:1111", "sha256:2222"}
        assert pins[1].marker == 'platform_machine != "s390x"'
        assert pins[1].hashes == {"sha256:3333"}

    def test_parses_extras_and_local_versions(self, tmp_path: Path):
        """Handles extras in pin names and local version identifiers."""
        requirements_txt = _write_lockfile(
            tmp_path,
            "autogluon-tabular[fastai]==1.6.3+rhaiv.2 \\\n    --hash=sha256:abc\n",
        )

        pins = parse_lockfile_pins(requirements_txt)

        assert len(pins) == 1
        assert pins[0].name == "autogluon-tabular"
        assert pins[0].version == "1.6.3+rhaiv.2"
        assert pins[0].display == "autogluon-tabular[fastai]==1.6.3+rhaiv.2"

    def test_skips_option_and_comment_lines(self, tmp_path: Path):
        """Returns no pins for lockfiles containing only options and comments."""
        requirements_txt = _write_lockfile(
            tmp_path,
            "--index-url https://example.com/simple\n# comment\n\n    # via -r requirements.in\n",
        )

        assert parse_lockfile_pins(requirements_txt) == []


class TestFileArch:
    """Tests for file_arch filename classification."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("pillow-13.0.0-1-cp312-cp312-linux_ppc64le.whl", "ppc64le"),
            ("xgboost-3.3.0-1-py3-none-linux_aarch64.whl", "aarch64"),
            ("torch-2.13.0-3-cp312-cp312-linux_s390x.whl", "s390x"),
            ("ray-2.58.0-1-cp312-cp312-linux_x86_64.whl", "x86_64"),
            ("numpy-2.1.0-cp312-cp312-manylinux_2_28_x86_64.whl", "x86_64"),
            ("autogluon_tabular-1.6.3+rhaiv.2-1-py3-none-any.whl", ARCH_LESS),
            ("rapidocr-3.9.2-2-py3-none-any.whl", ARCH_LESS),
            ("six-1.17.0-py2.py3-none-any.whl", ARCH_LESS),
            ("numpy-2.1.0-cp312-cp312-win_amd64.whl", OTHER_PLATFORM),
            ("six-1.17.0.tar.gz", SOURCE),
            ("autogluon_tabular-1.6.3+rhaiv.2.tar.gz", SOURCE),
            ("requests-2.32.3.zip", SOURCE),
            ("not-a-wheel.txt", OTHER_PLATFORM),
        ],
    )
    def test_classifies_filenames(self, filename: str, expected: str):
        """Classifies RHAI wheel, arch-less wheel, sdist, and unknown filenames."""
        assert file_arch(filename) == expected


class TestAllowedArches:
    """Tests for marker-based architecture filtering."""

    def test_no_marker_allows_all_architectures(self):
        """Expects every supported architecture when a pin has no marker."""
        assert allowed_arches(None) == set(LINUX_ARCHES)
        assert allowed_arches("") == set(LINUX_ARCHES)

    def test_not_marker_excludes_single_architecture(self):
        """platform_machine != 's390x' excludes only s390x."""
        assert allowed_arches('platform_machine != "s390x"') == {"x86_64", "aarch64", "ppc64le"}

    def test_or_marker_keeps_listed_architectures(self):
        """An or-marker keeps only the architectures it lists."""
        assert allowed_arches('platform_machine == "x86_64" or platform_machine == "aarch64"') == {"x86_64", "aarch64"}

    def test_non_platform_marker_keeps_all_architectures(self):
        """Markers without platform_machine evaluate identically on every architecture."""
        assert allowed_arches('python_version >= "3.10"') == set(LINUX_ARCHES)

    def test_marker_allows_arch_evaluates_single_architecture(self):
        """Evaluates a marker for one machine architecture."""
        assert marker_allows_arch('platform_machine != "s390x"', "s390x") is False
        assert marker_allows_arch('platform_machine != "s390x"', "x86_64") is True


class TestVerifyLockfileCoverage:
    """Tests for verify_lockfile_coverage."""

    def test_complete_arch_package_passes(self, tmp_path: Path):
        """A package with all four architectures hashed raises no findings."""
        files = _pillow_files()
        requirements_txt = _write_lockfile(tmp_path, f"pillow==13.0.0 \\\n{_pillow_hashes(files)}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"pillow": files}))

        assert findings == []

    def test_missing_arch_hash_is_reported(self, tmp_path: Path):
        """Detects a lockfile missing the hash of a wheel published on the index."""
        files = _pillow_files()
        ppc64le = files[-1]
        requirements_txt = _write_lockfile(tmp_path, f"pillow==13.0.0 \\\n{_pillow_hashes(files, skip=ppc64le)}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"pillow": files}))

        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert findings[0].package == "pillow==13.0.0"
        assert ppc64le.filename in findings[0].message

    def test_arch_less_package_requires_all_builds_hashed(self, tmp_path: Path):
        """Arch-less wheels of every build tag must be hashed in the lockfile."""
        files = [
            _index_file("rapidocr-3.9.2-1-py3-none-any.whl"),
            _index_file("rapidocr-3.9.2-2-py3-none-any.whl"),
        ]
        requirements_txt = _write_lockfile(tmp_path, f"rapidocr==3.9.2 \\\n    --hash=sha256:{files[0].sha256}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"rapidocr": files}))

        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert files[1].filename in findings[0].message

    def test_sdist_hash_is_required(self, tmp_path: Path):
        """Sdists published on the index must be hashed as well."""
        files = [
            _index_file("six-1.17.0-py2.py3-none-any.whl"),
            _index_file("six-1.17.0.tar.gz"),
        ]
        requirements_txt = _write_lockfile(tmp_path, f"six==1.17.0 \\\n    --hash=sha256:{files[0].sha256}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"six": files}))

        assert len(findings) == 1
        assert files[1].filename in findings[0].message

    def test_marker_skips_excluded_architecture_wheels(self, tmp_path: Path):
        """Wheels for marker-excluded architectures are neither required nor reported."""
        files = [
            _index_file("triton-3.6.0-4-cp312-cp312-linux_x86_64.whl"),
            _index_file("triton-3.6.0-4-cp312-cp312-linux_aarch64.whl"),
            _index_file("triton-3.6.0-4-cp312-cp312-linux_ppc64le.whl"),
            _index_file("triton-3.6.0-4-cp312-cp312-linux_s390x.whl"),
        ]
        marker = 'platform_machine != "s390x"'
        hashes = "\n".join(
            f"    --hash=sha256:{index_file.sha256} \\" for index_file in files if index_file.arch != "s390x"
        )
        requirements_txt = _write_lockfile(tmp_path, f"triton==3.6.0 ; {marker} \\\n{hashes}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"triton": files}))

        assert findings == []

    def test_marker_narrows_expected_architectures(self, tmp_path: Path):
        """A marker restricting architectures tolerates the index lacking the others."""
        files = [
            _index_file("ray-2.58.0-1-cp312-cp312-linux_x86_64.whl"),
            _index_file("ray-2.58.0-1-cp312-cp312-linux_aarch64.whl"),
        ]
        marker = 'platform_machine == "x86_64" or platform_machine == "aarch64"'
        requirements_txt = _write_lockfile(
            tmp_path,
            f"ray==2.58.0 ; {marker} \\\n{_pillow_hashes(files)}\n",
        )

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"ray": files}))

        assert findings == []

    def test_index_missing_architectures_is_an_error_by_default(self, tmp_path: Path):
        """A package with arch-specific wheels but fewer than four architectures fails."""
        files = [
            _index_file("xgboost-3.3.0-1-py3-none-linux_x86_64.whl"),
            _index_file("xgboost-3.3.0-1-py3-none-linux_aarch64.whl"),
        ]
        requirements_txt = _write_lockfile(tmp_path, f"xgboost==3.3.0 \\\n{_pillow_hashes(files)}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"xgboost": files}))

        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "ppc64le" in findings[0].message
        assert "s390x" in findings[0].message

    def test_index_missing_architectures_can_warn(self, tmp_path: Path):
        """arch_check='warn' downgrades index-side architecture gaps to warnings."""
        files = [
            _index_file("xgboost-3.3.0-1-py3-none-linux_x86_64.whl"),
            _index_file("xgboost-3.3.0-1-py3-none-linux_aarch64.whl"),
        ]
        requirements_txt = _write_lockfile(tmp_path, f"xgboost==3.3.0 \\\n{_pillow_hashes(files)}\n")

        findings = verify_lockfile_coverage(
            requirements_txt, client=FakeIndexClient({"xgboost": files}), arch_check="warn"
        )

        assert len(findings) == 1
        assert findings[0].severity == "warning"

    def test_arch_less_packages_have_no_architecture_expectation(self, tmp_path: Path):
        """Pure packages publishing fewer than four architectures do not fail."""
        files = [_index_file("rapidocr-3.9.2-1-py3-none-any.whl")]
        requirements_txt = _write_lockfile(tmp_path, f"rapidocr==3.9.2 \\\n    --hash=sha256:{files[0].sha256}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"rapidocr": files}))

        assert findings == []

    def test_stale_hash_is_reported(self, tmp_path: Path):
        """Detects lockfile hashes that match no file published on the index."""
        files = _pillow_files()
        requirements_txt = _write_lockfile(
            tmp_path, f"pillow==13.0.0 \\\n{_pillow_hashes(files)}\n    --hash=sha256:deadbeef\n"
        )

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"pillow": files}))

        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "deadbeef" in findings[0].message

    def test_no_files_for_pinned_version_is_reported(self, tmp_path: Path):
        """Detects pins whose version no longer has files on the index."""
        files = [
            _index_file("pillow-12.3.0-2-cp312-cp312-linux_x86_64.whl"),
        ]
        requirements_txt = _write_lockfile(tmp_path, f"pillow==13.0.0 \\\n    --hash=sha256:{files[0].sha256}\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"pillow": files}))

        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "no files" in findings[0].message

    def test_multiple_build_tags_all_required(self, tmp_path: Path):
        """Every build tag published for an architecture must be hashed."""
        files = [
            _index_file(f"torchvision-0.28.0-{build}-cp312-cp312-linux_{arch}.whl")
            for build in (2, 3)
            for arch in ("x86_64", "aarch64", "ppc64le", "s390x")
        ]
        requirements_txt = _write_lockfile(
            tmp_path, f"torchvision==0.28.0 \\\n{_pillow_hashes(files, skip=files[1])}\n"
        )

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"torchvision": files}))

        assert len(findings) == 1
        assert files[1].filename in findings[0].message

    def test_fetch_error_becomes_error_finding(self, tmp_path: Path):
        """Index lookup failures surface as error findings instead of aborting."""
        requirements_txt = _write_lockfile(tmp_path, "pillow==13.0.0 \\\n    --hash=sha256:1111\n")

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient(error="boom"))

        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert "could not fetch" in findings[0].message

    def test_input_markers_are_used_when_pin_has_none(self, tmp_path: Path):
        """Falls back to requirements.in markers for pins without a marker."""
        files = [
            _index_file("triton-3.6.0-4-cp312-cp312-linux_x86_64.whl"),
            _index_file("triton-3.6.0-4-cp312-cp312-linux_aarch64.whl"),
            _index_file("triton-3.6.0-4-cp312-cp312-linux_ppc64le.whl"),
        ]
        requirements_txt = _write_lockfile(tmp_path, f"triton==3.6.0 \\\n{_pillow_hashes(files)}\n")

        findings = verify_lockfile_coverage(
            requirements_txt,
            client=FakeIndexClient({"triton": files}),
            input_markers={"triton": 'platform_machine != "s390x"'},
        )

        assert findings == []

    def test_invalid_marker_is_reported(self, tmp_path: Path):
        """Invalid marker syntax surfaces as an error finding."""
        files = _pillow_files(with_ppc64le=False)
        requirements_txt = _write_lockfile(
            tmp_path,
            f"pillow==13.0.0 ; platform_machine == \\\n{_pillow_hashes(files)}\n",
        )

        findings = verify_lockfile_coverage(requirements_txt, client=FakeIndexClient({"pillow": files}))

        assert any("invalid environment marker" in finding.message for finding in findings)

    def test_rejects_unsupported_arch_check_mode(self, tmp_path: Path):
        """Raises ValueError for an unknown arch check mode."""
        requirements_txt = _write_lockfile(tmp_path, "pillow==13.0.0\n")

        with pytest.raises(ValueError, match="Unsupported arch check mode"):
            verify_lockfile_coverage(requirements_txt, client=FakeIndexClient(), arch_check="boom")


class TestSimpleIndexClient:
    """Tests for SimpleIndexClient against mocked HTTP responses."""

    @staticmethod
    def _response(payload: dict) -> mock.MagicMock:
        """Build a mocked urlopen return value serving a JSON payload."""
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
        return response

    def test_fetches_and_caches_pep691_metadata(self):
        """Parses PEP 691 JSON responses and caches them per package."""
        payload = {
            "meta": {"api-version": "1.1"},
            "name": "pillow",
            "files": [
                {
                    "filename": "pillow-13.0.0-1-cp312-cp312-linux_x86_64.whl",
                    "hashes": {"sha256": "a" * 64},
                },
                {"filename": "pillow-12.3.0-2-cp312-cp312-linux_s390x.whl"},
            ],
        }
        client = SimpleIndexClient("https://example.com/simple/")

        with mock.patch("urllib.request.urlopen", return_value=self._response(payload)) as urlopen_mock:
            files = client.files("Pillow")
            again = client.files("pillow")

        assert urlopen_mock.call_count == 1
        request = urlopen_mock.call_args.args[0]
        assert request.full_url == "https://example.com/simple/pillow/"
        assert request.headers.get("Accept") == "application/vnd.pypi.simple.v1+json"
        assert files == again
        assert [index_file.filename for index_file in files] == [
            "pillow-13.0.0-1-cp312-cp312-linux_x86_64.whl",
            "pillow-12.3.0-2-cp312-cp312-linux_s390x.whl",
        ]
        assert files[0].sha256 == "a" * 64
        assert files[0].arch == "x86_64"
        assert files[1].sha256 is None

    def test_wraps_http_errors(self):
        """Network failures raise IndexLookupError with the request URL."""
        client = SimpleIndexClient("https://example.com/simple")

        with (
            mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")),
            mock.patch("time.sleep"),  # skip retry backoff
        ):
            with pytest.raises(IndexLookupError, match="example.com/simple/pillow"):
                client.files("pillow")

    def test_retries_transient_network_errors(self):
        """A failing first fetch attempt is retried before giving up."""
        payload = {"meta": {}, "files": []}
        client = SimpleIndexClient("https://example.com/simple")
        failures = [urllib.error.URLError("dns hiccup"), None]

        def flaky_urlopen(_request, timeout):  # noqa: ARG001
            failure = failures.pop(0)
            if failure is not None:
                raise failure
            return self._response(payload)

        with (
            mock.patch("urllib.request.urlopen", side_effect=flaky_urlopen),
            mock.patch("time.sleep"),
        ):
            files = client.files("pillow")

        assert files == []
        assert failures == []  # both attempts were consumed

    def test_wraps_non_json_responses(self):
        """HTML responses (JSON unsupported) raise IndexLookupError."""
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(b"<html>not json</html>")
        client = SimpleIndexClient("https://example.com/simple")

        with mock.patch("urllib.request.urlopen", return_value=response):
            with pytest.raises(IndexLookupError, match="pillow"):
                client.files("pillow")
