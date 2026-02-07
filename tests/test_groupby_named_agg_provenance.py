# tests/test_groupby_named_agg_provenance.py
import pandas as pd
import pytest
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)


def _aux_tokens(decoded_df, group_key) -> set[str]:
    prov_cell = decoded_df.loc[group_key, "_prov"]

    tokens = set()
    for item in prov_cell:
        if isinstance(item, list):
            tokens.update(item)
        else:
            tokens.add(item)

    return {p for p in tokens if p.startswith("aux_table:")}


def test_T06_groupby_named_agg_preserves_provenance(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    df = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy"],
            "City": ["Rome", "Brussels", "Palermo"],
            "population": [1000, 2000, 1500],
        }
    )

    t = skrub.var("aux_table", df)

    out = t.groupby("Country").agg(
        total_population=("population", "sum"),
        city_count=("City", "nunique"),
    )

    preview = out.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    assert decoded.loc["Italy", "total_population"] == 2500
    assert decoded.loc["Belgium", "total_population"] == 2000
    assert decoded.loc["Italy", "city_count"] == 2
    assert decoded.loc["Belgium", "city_count"] == 1

    assert _aux_tokens(decoded, "Italy") == {"aux_table:0", "aux_table:2"}
    assert _aux_tokens(decoded, "Belgium") == {"aux_table:1"}
