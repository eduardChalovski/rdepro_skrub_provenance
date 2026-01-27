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


def test_T10_left_merge_provenance_with_nulls(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    left_df = pd.DataFrame(
        {
            "Country": ["USA", "Italy", "Georgia"],
            "L": [1, 2, 3],
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

    out = left.merge(right, on="Country", how="left")

    preview = out.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    assert len(decoded) == 3, "Left join should preserve all left rows."

    usa_row = decoded[decoded["Country"] == "USA"].iloc[0]
    geo_row = decoded[decoded["Country"] == "Georgia"].iloc[0]
    it_row = decoded[decoded["Country"] == "Italy"].iloc[0]

    assert pd.isna(usa_row["City"])
    assert pd.isna(geo_row["City"])
    assert it_row["City"] == "Rome"

    usa_tokens = _flatten_tokens(usa_row["_prov"])
    geo_tokens = _flatten_tokens(geo_row["_prov"])
    it_tokens = _flatten_tokens(it_row["_prov"])

    assert any(t.startswith("left_table:") for t in usa_tokens)
    assert any(t.startswith("left_table:") for t in geo_tokens)
    assert any(t.startswith("left_table:") for t in it_tokens)

    assert not any(t.startswith("right_table:") for t in usa_tokens), (
        f"USA has no match, should not contain right_table provenance. Got: {usa_tokens}"
    )
    assert not any(t.startswith("right_table:") for t in geo_tokens), (
        f"Georgia has no match, should not contain right_table provenance. Got: {geo_tokens}"
    )
    assert any(t.startswith("right_table:") for t in it_tokens), (
        f"Italy matches, should contain right_table provenance. Got: {it_tokens}"
    )
