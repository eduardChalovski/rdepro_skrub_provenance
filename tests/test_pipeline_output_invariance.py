import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
)


SENTINEL = -1 


def _normalize_for_equals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out.fillna(SENTINEL)


def _is_prov_col(col) -> bool:
    """
    col could be str or tuple (MultiIndex).
    """
    if isinstance(col, tuple):
        s = " ".join(map(str, col)).lower()
    else:
        s = str(col).lower()
    return ("_prov" in s) or ("provenance" in s) or (s.endswith("prov")) or (" prov" in s) or ("prov" in s)


def _drop_prov_cols(df: pd.DataFrame) -> pd.DataFrame:
    prov_cols = [c for c in df.columns if _is_prov_col(c)]
    return df.drop(columns=prov_cols, errors="ignore")


def _run_transformation(table):
    out = table.groupby("Country").agg(["sum", "max"])
    return out.skb.preview()


def test_pipeline_output_invariance(restore_skrub_monkeypatch):
    df = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy"],
            "population": [1000, 2000, 1500],
        }
    )


    t0 = skrub.var("aux_table", df)
    out_no_prov = _run_transformation(t0)

    enable_why_data_provenance()
    t1 = skrub.var("aux_table", df)
    out_with_prov = _run_transformation(t1)

    
    evaluated = evaluate_provenance_fast(out_with_prov)
    
    assert any(_is_prov_col(c) for c in evaluated.columns), "Expected provenance columns when provenance is enabled"

   
    stripped = _drop_prov_cols(out_with_prov)

    a = _normalize_for_equals(out_no_prov)
    b = _normalize_for_equals(stripped)

    assert list(a.columns) == list(b.columns)
    assert a.equals(b)
