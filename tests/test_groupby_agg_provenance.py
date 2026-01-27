# tests/test_groupby_agg_provenance.py
import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)

def _prov_as_set_of_strings(decoded_df, row_key) -> set[str]:
    """
    Extract a row's decoded provenance as a set of normalized tokens.
    """
    prov_list = decoded_df.loc[row_key, "_prov"]
    return set(prov_list)


def test_T05_groupby_agg_dict_provenance_union(restore_skrub_monkeypatch):
    """
    T05 — groupby + dict-style aggregation must UNION provenance of contributing rows.

        - `df_main` contains 3 countries: USA, Italy, Belgium
        - `df_aux` contains 3 rows:
            * Italy (row 0)  population=1000
            * Belgium (row 1) population=2000
            * Italy (row 2)  population=1500
        - We left-join `main` with `aux` on Country.
        - Then we compute: groupby("Country").agg({"population": "sum"}).

        1) Correct aggregated values:
            Italy   -> 1000 + 1500
            Belgium -> 2000
        2) Correct provenance behavior for the aggregated rows:
            The output provenance for each group must be the UNION of all source
            tuples that contributed to the aggregate.
            Concretely, for Italy we require both aux rows (0 and 2),
            and for Belgium we require aux row (1).
    """
    enable_why_data_provenance()
    df_main = pd.DataFrame({"Country": ["USA", "Italy", "Belgium"]})

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

    out = joined.groupby("Country").agg({"population": "sum"})

    preview = out.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    assert "population" in decoded.columns
    assert "_prov" in decoded.columns

    assert "Italy" in decoded.index
    assert "Belgium" in decoded.index

    assert decoded.loc["Italy", "population"] == 1000 + 1500
    assert decoded.loc["Belgium", "population"] == 2000

    italy_prov = _prov_as_set_of_strings(decoded, "Italy")
    belgium_prov = _prov_as_set_of_strings(decoded, "Belgium")

    italy_aux = {p for p in italy_prov if p.startswith("aux_table:")}
    belgium_aux = {p for p in belgium_prov if p.startswith("aux_table:")}

    assert italy_aux == {"aux_table:0", "aux_table:2"}, (
        f"Incorrect provenance for Italy. Expected aux rows {{0,2}}, got: {sorted(italy_aux)}"
    )
    assert belgium_aux == {"aux_table:1"}, (
        f"Incorrect provenance for Belgium. Expected aux row {{1}}, got: {sorted(belgium_aux)}"
    )
