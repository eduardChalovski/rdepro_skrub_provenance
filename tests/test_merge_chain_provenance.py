# tests/test_merge_chain_provenance.py
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


def test_T12_merge_chain_provenance_composition(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    left_df = pd.DataFrame(
        {
            "Country": ["USA", "Italy", "Georgia"],
            "L": [1, 2, 3],
        }
    )
    aux_df = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium"],
            "City": ["Rome", "Brussels"],
            "population": [1000, 2000],
        }
    )
    people_df = pd.DataFrame(
        {
            "Country": ["Italy", "Germany"],
            "Name": ["Person1", "Person2"],
        }
    )

    left = skrub.var("left_table", left_df)
    aux = skrub.var("aux_table", aux_df)
    people = skrub.var("people_table", people_df)

    joined1 = left.merge(aux, on="Country", how="inner")          # keeps Italy only
    joined2 = joined1.merge(people, on="Country", how="left")     # keeps Italy, may add Name

    preview = joined2.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    assert len(decoded) == 1, "After inner then left merge, only Italy should remain."

    row = decoded.iloc[0]
    assert row["Country"] == "Italy"
    assert row["City"] == "Rome"
    assert row["Name"] == "Person1"

    tokens = _flatten_tokens(row["_prov"])

    # Provenance should include all three sources for the Italy row
    assert any(t.startswith("left_table:") for t in tokens), f"Missing left_table provenance: {tokens}"
    assert any(t.startswith("aux_table:") for t in tokens), f"Missing aux_table provenance: {tokens}"
    assert any(t.startswith("people_table:") for t in tokens), f"Missing people_table provenance: {tokens}"
