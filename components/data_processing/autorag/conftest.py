import subprocess
from pathlib import Path

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_collection(session):
    """Wraps pytest_collection hook with custom logic to be done beforehand.

    In order to properly collect unit tests for functions nested within kfp.component-wrapped functions,
    these nested functions have to be dynamically extracted to a separate file
    (so that they can be imported in respective unit tests).

    Notes:
        This hook only executes for "initial" conftest files so the `component/training/autorag` path
        (or any of its subdirs) MUST be specified as pytest test_path.
    """
    for path in Path(__file__).parent.iterdir():
        if not path.is_dir():  # we're looking only for kfp component directories
            continue
        if (component_path := (path / "component.py")).exists():
            if (nested_names_dest_path := path / "tests" / "nested_names.py").exists():
                continue
            command = (
                "/bin/bash "
                "components/training/autorag/rag_templates_optimization/tests/scripts/orchestrate_extraction.sh "
                f"{nested_names_dest_path.absolute()} {component_path.absolute()}"
            )
            subprocess.run(command.split(), shell=False, check=True)

        # If the outcome is an exception, will raise the exception.
    return (yield)


def pytest_addoption(parser, pluginmanager):
    """Registers an optional autorag-related command line option.

    Notes:
        This hook only executes for "initial" conftest files so the `component/training/autorag` path
        (or any of its subdirs) MUST be specified as pytest test_path.
    """
    # the same might have already be performed as part of `components.training.autorag` module
    training_autorag_conftest_path = Path("components", "training", "autorag", "conftest.py").absolute()
    # pdb.set_trace()
    if pluginmanager.hasplugin(str(training_autorag_conftest_path)):
        return

    parser.addoption(
        "--autorag-no-session-cleanup",
        action="store_true",
        dest="autorag_no_session_cleanup",
        help="Disable running the `session_cleanup` fixture. Useful for debugging or development works.",
    )


@pytest.fixture(scope="session", autouse=True)
def session_cleanup(request):
    """Performs cleanup after pytest session finishes in a way it:

        - deletes dynamically created (during pytest_collection) `nested_names.py` files

    The cleanup can be disabled (e.g. for debugging and other dev-friendly reasons) using the
    pytest cli option `--autorag-no-session-cleanup`
    """
    yield

    if request.config.getoption("autorag_no_session_cleanup", False):
        return

    if not __spec__.has_location:
        print(
            "Cannot determine module's loadable location. "
            "The cleanup will not run so some dynamically created artefacts may preserve."
        )
        return

    for path in Path(__spec__.origin).parent.iterdir():
        if not path.is_dir():  # we're looking only for kfp component directories
            continue
        if (nested_names_path := (path / "tests" / "nested_names.py")).exists():
            nested_names_path.unlink()
