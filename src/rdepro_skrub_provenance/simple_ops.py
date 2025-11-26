from __future__ import annotations
import pandas as pd


def rename_column(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """
    Rename a column in a pandas DataFrame.
    """
    return df.rename(columns={old: new}).copy()


def rename_column_with_prov(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """
    Rename a column while preserving the provenance column `_prov`.

    According to provenance rule R1 (Perm), renaming does not change the tuple provenance.
    The `_prov` column is therefore kept unchanged.
    """
    if "_prov" not in df.columns:
        raise ValueError("Missing provenance column '_prov' in input dataframe.")

    df2 = df.rename(columns={old: new}).copy()
    return df2
