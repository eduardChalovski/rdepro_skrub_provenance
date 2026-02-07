# tests/test_named_aggregation_provenance.py
import pandas as pd
import pytest
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)

@pytest.mark.xfail(reason="Named aggregation style may not be supported yet by provenance_agg patch.")
def test_T06_groupby_named_agg_provenance(restore_skrub_monkeypatch):
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

    # Value assertions
    assert decoded.loc["Italy", "total_population"] == 2500
    assert decoded.loc["Belgium", "total_population"] == 2000
    assert decoded.loc["Italy", "city_count"] == 2
    assert decoded.loc["Belgium", "city_count"] == 1

    # Provenance sanity: should contain aux_table rows
    italy_prov = set(decoded.loc["Italy", "_prov"])
    assert any(p.startswith("aux_table:") for p in italy_prov)
