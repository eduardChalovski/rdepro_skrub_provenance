# tests/test_decode_prov_column.py
import pandas as pd
import skrub

from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    decode_prov_column,
)


def test_T15_decode_prov_column_preserves_data_columns(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    df = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy"],
            "population": [1000, 2000, 1500],
        }
    )

    t = skrub.var("aux_table", df)

    preview = t.skb.preview()

    decoded = decode_prov_column(preview, evaluate_provenance_first=True)

    assert decoded["Country"].tolist() == df["Country"].tolist()
    assert decoded["population"].tolist() == df["population"].tolist()

    assert "_prov" in decoded.columns

    for cell in decoded["_prov"]:
        assert isinstance(cell, list), f"Expected list of decoded tokens, got: {type(cell)}"
        assert any(isinstance(x, str) and x.startswith("aux_table:") for x in cell), (
            f"Decoded provenance does not reference aux_table as expected: {cell}"
        )
