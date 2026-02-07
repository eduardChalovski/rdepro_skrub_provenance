import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
)


def _unwrap_depth(func) -> int:
    """
    Return how many times a callable has been wrapped (via `functools.wraps`).
    """
    depth = 0
    cur = func
    while hasattr(cur, "__wrapped__"):
        depth += 1
        cur = cur.__wrapped__
    return depth


def test_T01_patch_activated_and_idempotent(restore_skrub_monkeypatch):
    """
    T01 — Patch activation + idempotence.

    What this test checks:
        1) Activation:
           Calling `enable_why_data_provenance()` must monkey-patch Skrub internals
           by wrapping (decorating) specific functions (e.g., `evaluate` and `Var.compute`).
           We verify this by checking that the wrapping depth increases by exactly +1.

        2) Idempotence:
           Calling `enable_why_data_provenance()` a second time must NOT wrap again.
           In other words, the patching operation should be safe to call multiple times
           without accumulating nested wrappers (which would cause duplicated work,
           performance regressions, or incorrect provenance duplication).

    Why it matters:
        Monkey-patching is global process-wide state. If it is not idempotent, test
        suites and notebooks that re-import / re-run cells can silently break.
    """
    import skrub._data_ops._evaluation as eval_mod
    Var = skrub._data_ops._data_ops.Var

    # Before patching: expected wrapping depth is 0 (or whatever baseline is present).
    before_eval_depth = _unwrap_depth(eval_mod.evaluate)
    before_compute_depth = _unwrap_depth(Var.compute)

    # Activate patching.
    enable_why_data_provenance()

    # After patching: expected wrapping depth is baseline + 1.
    after_eval_depth = _unwrap_depth(eval_mod.evaluate)
    after_compute_depth = _unwrap_depth(Var.compute)

    assert after_eval_depth == before_eval_depth + 1, (
        "evaluate() was not patched as expected (unexpected wrapping depth)."
    )
    assert after_compute_depth == before_compute_depth + 1, (
        "Var.compute was not patched as expected (unexpected wrapping depth)."
    )

    # Idempotence: calling again must not increase wrapping depth further.
    enable_why_data_provenance()

    after2_eval_depth = _unwrap_depth(eval_mod.evaluate)
    after2_compute_depth = _unwrap_depth(Var.compute)

    assert after2_eval_depth == after_eval_depth, (
        "enable_why_data_provenance() is not idempotent: evaluate() was wrapped multiple times."
    )
    assert after2_compute_depth == after_compute_depth, (
        "enable_why_data_provenance() is not idempotent: Var.compute was wrapped multiple times."
    )


def test_T02_var_adds_provenance_columns(restore_skrub_monkeypatch):
    """
    T02 — Functional behavior: Var materialization must include provenance columns.

    What this test checks:
        After patch activation, building a Skrub `Var` from a DataFrame and forcing
        materialization (via `.skb.preview()`) must produce a DataFrame that contains
        at least one provenance column (columns starting with `_prov`).

        Additionally, each provenance cell must be "non-empty":
          - no None/NaN values,
          - if the cell is a list, the list must not be empty.

    Why it matters:
        This is an end-to-end check from the user's perspective: the patch must not
        only be installed, it must also *change the resulting data* as expected
        (i.e., provenance is actually attached to rows/values).
    """
    enable_why_data_provenance()

    df = pd.DataFrame(
        {
            "Country": ["USA", "Italy", "Belgium"],
            "x": [1, 2, 3],
        }
    )

    v = skrub.var("main_table", df)
    preview = v.skb.preview()  # Force computation / materialization.

    prov_cols = [c for c in preview.columns if c.startswith("_prov")]
    assert prov_cols, (
        "No `_prov*` column found after `.skb.preview()`. "
        "Provenance was likely not added at the `Var.compute` level."
    )

    # Ensure provenance is present and non-empty for every row.
    for c in prov_cols:
        col = preview[c]

        assert col.notna().all(), (
            f"Provenance column {c} contains NaN/None values."
        )

        def is_non_empty(cell) -> bool:
            """
            Accept common provenance encodings:
              - list: must be non-empty
              - int/str/other scalar: considered non-empty by definition
            """
            if isinstance(cell, list):
                return len(cell) > 0
            return True

        assert col.map(is_non_empty).all(), (
            f"Provenance column {c} contains at least one empty provenance value (e.g., [])."
        )

