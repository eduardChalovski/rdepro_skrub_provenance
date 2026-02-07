# tests/test_isin_filter_provenance.py
import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)


def _aux_prov(decoded_df, row_idx) -> set[str]:
    """
    Extract provenance tokens coming from aux_table for a given row.
    """
    return {
        p
        for p in decoded_df.loc[row_idx, "_prov"]
        if p.startswith("aux_table:")
    }


def test_T04_isin_filter_preserves_provenance(restore_skrub_monkeypatch):
    enable_why_data_provenance()
    df_main = pd.DataFrame(
        {
            "Country": ["USA", "Italy", "Georgia", "Belgium"],
        }
    )

    df_aux = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy", "USA"],
            "City": ["Palermo", "Brussels", "Rome", "NYC"],
        }
    )

    main = skrub.var("main_table", df_main)
    aux = skrub.var("aux_table", df_aux)

    joined = main.merge(aux, on="Country", how="inner")

    before = joined.skb.preview()
    before_eval = evaluate_provenance_fast(before)
    before_decoded = decode_prov_column(before_eval, evaluate_provenance_first=False)

    european_countries = ["Italy", "Belgium"]
    filtered = joined[joined["Country"].isin(european_countries)]

    after = filtered.skb.preview()
    after_eval = evaluate_provenance_fast(after)
    after_decoded = decode_prov_column(after_eval, evaluate_provenance_first=False)

    assert set(after_decoded["Country"].unique()) == {"Italy", "Belgium"}
    assert len(after_decoded) == 3  

    for _, row in after_decoded.iterrows():
        country = row["Country"]
        city = row["City"]

        matching_before = before_decoded[
            (before_decoded["Country"] == country)
            & (before_decoded["City"] == city)
        ]

        assert len(matching_before) == 1, (
            "Expected exactly one matching row before filtering."
        )

        prov_before = set(matching_before.iloc[0]["_prov"])
        prov_after = set(row["_prov"])

        assert prov_before == prov_after, (
            f"Provenance changed after isin filter for ({country}, {city}).\n"
            f"Before: {prov_before}\nAfter:  {prov_after}"
        )

    for idx in after_decoded.index:
        aux_tokens = _aux_prov(after_decoded, idx)
        assert aux_tokens, (
            "Filtered row lost aux_table provenance after isin filtering."
        )
