from __future__ import annotations
import pandas as pd


def with_provenance(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Add a provenance column `_prov` to the dataframe.

    Each row receives a unique provenance identifier derived from the
    source table name and the row index. This corresponds to the R1 rule
    in Perm: provenance initialization at the leaves of the data pipeline.
    """
    df = df.copy()
    df["_prov"] = [f"{source_name}:{i}" for i in range(len(df))]
    return df
