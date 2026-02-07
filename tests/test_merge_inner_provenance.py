# tests/test_merge_inner_provenance.py
import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)


def _aux_prov_set(decoded_df, row_idx) -> set[str]:
    """
    Extract the auxiliary-table provenance tokens for a given output row.
    """
    return {
        p
        for p in decoded_df.loc[row_idx, "_prov"]
        if p.startswith("aux_table:")
    }


def test_T03_merge_inner_provenance(restore_skrub_monkeypatch):
    """
    T03 — INNER merge must preserve row-level provenance from the matching aux row.

        - `main_table` contains: USA, Italy, Belgium
        - `aux_table` contains:
            row 0: Italy   -> Palermo
            row 1: Belgium -> Brussels
            row 2: Italy   -> Rome
        - We compute: main.merge(aux, on="Country", how="inner")

        - USA has no match in aux => removed by INNER join
        - Italy matches two aux rows => appears twice
        - Belgium matches one aux row => appears once
        => total output rows = 3

        1) Join cardinality and output content are correct.
        2) Provenance for each output row includes exactly one aux_table contributor:
            - the Belgium output row must include {"aux_table:1"}
            - the two Italy output rows must include {"aux_table:0"} and {"aux_table:2"}
              (order between the two Italy rows may vary, so we compare as sets-of-sets).
    """
    enable_why_data_provenance()

    df_main = pd.DataFrame({"Country": ["USA", "Italy", "Belgium"]})
    df_aux = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy"],
            "City": ["Palermo", "Brussels", "Rome"],
        }
    )

    main = skrub.var("main_table", df_main)
    aux = skrub.var("aux_table", df_aux)

    joined = main.merge(aux, on="Country", how="inner")

    preview = joined.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    assert len(decoded) == 3, "Unexpected number of rows for INNER join."
    assert "_prov" in decoded.columns

    assert sorted(decoded["Country"].tolist()) == ["Belgium", "Italy", "Italy"]

    decoded = decoded.reset_index(drop=True)

    italy_rows = decoded[decoded["Country"] == "Italy"]
    belgium_rows = decoded[decoded["Country"] == "Belgium"]

    assert len(italy_rows) == 2
    assert len(belgium_rows) == 1

    italy_aux_sets = [
        _aux_prov_set(decoded, idx)
        for idx in italy_rows.index
    ]

    assert set(map(frozenset, italy_aux_sets)) == {frozenset({"aux_table:0"}), frozenset({"aux_table:2"})}, (
        f"Incorrect aux provenance for Italy rows. Got: {italy_aux_sets}"
    )

    belgium_idx = belgium_rows.index[0]
    belgium_aux = _aux_prov_set(decoded, belgium_idx)
    assert belgium_aux == {"aux_table:1"}, (
        f"Incorrect aux provenance for Belgium row. Got: {sorted(belgium_aux)}"
    )
