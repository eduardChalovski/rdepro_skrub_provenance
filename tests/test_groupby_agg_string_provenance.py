# tests/test_groupby_agg_string_provenance.py
import pandas as pd
import skrub

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
    evaluate_provenance_fast,
    decode_prov_column,
)


def _aux_tokens(decoded_df, group_key) -> set[str]:
    return {p for p in decoded_df.loc[group_key, "_prov"] if p.startswith("aux_table:")}


def test_T08_groupby_agg_string_sum_provenance(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    df = pd.DataFrame(
        {
            "Country": ["Italy", "Belgium", "Italy"],
            "population": [1000, 2000, 1500],
        }
    )

    t = skrub.var("aux_table", df)

    out = t.groupby("Country").agg("sum")

    preview = out.skb.preview()
    evaluated = evaluate_provenance_fast(preview)
    decoded = decode_prov_column(evaluated, evaluate_provenance_first=False)

    # ---- value assertions ----
    assert decoded.loc["Italy", "population"] == 2500
    assert decoded.loc["Belgium", "population"] == 2000

    # ---- provenance assertions (exact union) ----
    assert _aux_tokens(decoded, "Italy") == {"aux_table:0", "aux_table:2"}
    assert _aux_tokens(decoded, "Belgium") == {"aux_table:1"}
