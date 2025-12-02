from __future__ import annotations
import pandas as pd

PROV_COLUMN = "_prov"


def with_provenance(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Initialize provenance for a base relation.

    Each row receives a unique provenance identifier derived from the
    source table name and the row index. This corresponds to the leaf
    initialization rule in Perm (R0 in spirit): provenance initialization
    at the leaves of the data pipeline.
    """
    df = df.copy()
    df[PROV_COLUMN] = [f"{source_name}:{i}" for i in range(len(df))]
    return df

def _sum_provenances(provs: pd.Series) -> str:
    """
    Combine a group of provenance expressions using semiring addition (+).
    """
    provs = provs.astype(str)
    unique_provs = list(provs) 
    if not unique_provs:
        return ""
    if len(unique_provs) == 1:
        return unique_provs[0]
    return "(" + ") + (".join(unique_provs) + ")"


def _product_provenances(p1: str, p2: str) -> str:
    """
    Combine two provenance expressions using semiring multiplication (*).
    """
    p1 = str(p1)
    p2 = str(p2)
    return f"({p1})*({p2})"



def rename_relation_with_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """
    In relational algebra, renaming a relation changes only its symbolic
    name, not its content. Provenance is therefore left untouched.
    """
    if PROV_COLUMN not in df.columns:
        raise ValueError(f"Expected provenance column '{PROV_COLUMN}' in dataframe.")
    return df.copy()



