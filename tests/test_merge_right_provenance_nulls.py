# tests/test_merge_right_provenance_nulls.py
import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)


def _flatten_tokens(cell) -> set[str]:
    tokens = set()
    for item in cell:
        if isinstance(item, list):
            tokens.update(item)
        else:
            tokens.add(item)
    return tokens


def test_T11_right_merge_provenance_with_nulls(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    left_df = pd.DataFrame(
        {
            "Country": ["USA", "Italy"],
            "L": [1, 2],
        }
    )
    right_df = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium"],
            "City": ["Rome", "Brussels"],
            "R": [10, 20],
        }
    )

    left = skrub.var("left_table", left_df)
    right = skrub.var("right_table", right_df)

    out = left.merge(right, on="Country", how="right")

    preview = out.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    assert len(decoded) == 2, "Right join should preserve all right rows."

    it_row = decoded[decoded["Country"] == "Italy"].iloc[0]
    be_row = decoded[decoded["Country"] == "Belgium"].iloc[0]

    assert pd.isna(be_row["L"]), "Right-only row should have NaN for left columns."
    assert be_row["City"] == "Brussels"

    it_tokens = _flatten_tokens(it_row["_prov"])
    be_tokens = _flatten_tokens(be_row["_prov"])


    assert any(t.startswith("left_table:") for t in it_tokens)
    assert any(t.startswith("right_table:") for t in it_tokens)

   
    assert not any(t.startswith("left_table:") for t in be_tokens), (
        f"Belgium is right-only, should not contain left_table provenance. Got: {be_tokens}"
    )
    assert any(t.startswith("right_table:") for t in be_tokens), (
        f"Belgium is right-only, should contain right_table provenance. Got: {be_tokens}"
    )
