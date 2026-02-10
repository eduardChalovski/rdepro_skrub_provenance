# tests/test_DeterminismCheck.py
import numpy as np
import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
)

def _replace_nan_with_minus_one(x):
    if isinstance(x, (list, np.ndarray)):
        return [_replace_nan_with_minus_one(i) for i in x]
    return -1 if pd.isna(x) else x


def _normalize_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame comparable run-to-run:
    - fill scalar NaNs
    - normalize provenance columns that may contain lists with NaNs
    - sort columns to avoid ordering-related diffs
    """
    out = df.copy()
    out = out.fillna(-1)

    prov_cols = [c for c in out.columns if c.startswith("_prov")]
    for c in prov_cols:
        out[c] = out[c].apply(_replace_nan_with_minus_one)

    out = out.reindex(sorted(out.columns), axis=1)

    out = out.sort_index()

    return out


def _build_small_deterministic_dataops_graph():
    """
    Builds a small but representative DataOps graph:
    - two inputs
    - join
    - selection
    - groupby aggregation
    - projection (via skb.select to preserve provenance)
    """
    df_main = pd.DataFrame(
        {
            "Country": ["USA", "Italy", "Belgium", "Italy"],
            "order_id": [1, 2, 3, 4],
            "x": [10, 20, 30, 40],
        }
    )
    df_aux = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy"],
            "City": ["Palermo", "Brussels", "Rome"],
            "population": [1000, 2000, 1500],
        }
    )

    main = skrub.var("main_table", df_main)
    aux = skrub.var("aux_table", df_aux)

    joined = main.merge(aux, on="Country", how="left")

    filtered = joined[joined["x"] >= 20]

    agg = filtered.groupby("Country").agg({"population": "sum", "x": "mean"})

    agg = agg.skb.select(["population", "x"])

    return agg


def test_determinism_unit_style(restore_skrub_monkeypatch):
    """
    Determinism check (unit-test style):
    run the same DataOps pipeline twice and ensure the evaluated / normalized
    outputs are identical.
    """
    enable_why_data_provenance()

    # First run
    out1 = _build_small_deterministic_dataops_graph()
    df1 = out1.skb.preview()
    df1 = evaluate_provenance_fast(df1) 
    df1 = _normalize_for_comparison(df1)

    # Second run (fresh graph)
    out2 = _build_small_deterministic_dataops_graph()
    df2 = out2.skb.preview()
    df2 = evaluate_provenance_fast(df2)
    df2 = _normalize_for_comparison(df2)

    assert df1.equals(df2), "Non-deterministic: outputs differ between two identical runs"
