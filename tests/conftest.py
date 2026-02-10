# tests/conftest.py
import pytest
import skrub


def pytest_addoption(parser):
    parser.addoption(
        "--track-provenance",
        action="store_true",
        default=False,
        help="Enable provenance tracking during tests.",
    )


@pytest.fixture
def track_provenance(request) -> bool:
    return bool(request.config.getoption("--track-provenance"))


@pytest.fixture
def restore_skrub_monkeypatch():
    """
    keep and restore points patched by enable_why_data_provenance().
    """
    import skrub._data_ops._evaluation as eval_mod
    Var = skrub._data_ops._data_ops.Var

    original_evaluate = eval_mod.evaluate
    original_var_compute = Var.compute

    yield

    eval_mod.evaluate = original_evaluate
    Var.compute = original_var_compute
