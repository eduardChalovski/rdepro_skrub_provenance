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


#region Helpers

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


#region Renaming

def rename_relation_with_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """
    In relational algebra, renaming a relation changes only its symbolic
    name, not its content. Provenance is therefore left untouched.
    """
    if PROV_COLUMN not in df.columns:
        raise ValueError(f"Expected provenance column '{PROV_COLUMN}' in dataframe.")
    return df.copy()


#region  Projection

def project_with_provenance(df: pd.DataFrame, columns) -> pd.DataFrame:
    """
    Projection keeps a subset of columns but MUST preserve the provenance
    attribute. Concretely, we project onto the selected columns plus the
    `_prov` column. We do not remove duplicates; i.e., we use bag semantics
    by default (set semantics would require an additional `drop_duplicates`).
    """
    if PROV_COLUMN not in df.columns:
        raise ValueError(f"Expected provenance column '{PROV_COLUMN}' in dataframe.")

    cols = list(columns)
    if PROV_COLUMN not in cols:
        cols.append(PROV_COLUMN)

    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Projection refers to unknown columns: {missing}")

    return df[cols].copy()


#region Selection / Filter

def select_with_provenance(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """
    A selection (filter) does not modify provenance. It restricts rows
    but keeps their provenance value unchanged.
    """
    if PROV_COLUMN not in df.columns:
        raise ValueError(f"Expected provenance column '{PROV_COLUMN}' in dataframe.")

    if not isinstance(mask, pd.Series):
        raise TypeError("`mask` must be a pandas Series of booleans.")
    if len(mask) != len(df):
        raise ValueError("`mask` must have the same length as `df`.")

    return df[mask].copy()


#region Cartesian Product

def cartesian_product_with_provenance(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """
    The provenance of a tuple produced by combining two tuples is the
    *product* (combination) of the provenances of the two input tuples.

    We return all columns from both inputs (using pandas' default suffixing
    for overlapping column names) and a single `_prov` column computed as:

        _prov = (left._prov) * (right._prov)
    """
    if PROV_COLUMN not in left.columns or PROV_COLUMN not in right.columns:
        raise ValueError(f"Both inputs must contain a '{PROV_COLUMN}' column.")

    # Rename provenance columns temporarily to avoid name clashes
    l = left.copy().rename(columns={PROV_COLUMN: "_prov_left"})
    r = right.copy().rename(columns={PROV_COLUMN: "_prov_right"})

    # Cross join via dummy key
    l["_tmp_key"] = 1
    r["_tmp_key"] = 1
    joined = l.merge(r, on="_tmp_key").drop(columns="_tmp_key")

    # Combine provenance
    joined[PROV_COLUMN] = _product_provenances(
        joined["_prov_left"], joined["_prov_right"]
    )

    # Drop temporary provenance columns
    joined = joined.drop(columns=["_prov_left", "_prov_right"])

    return joined


# region Aggregation / GROUP BY

def groupby_aggregate_with_provenance(
    df: pd.DataFrame,
    by,
    agg_spec: dict,
) -> pd.DataFrame:
    """
    The provenance of an aggregated tuple is the *union* (semiring sum)
    of the provenances of all grouped tuples.

    We:
      1. Apply standard pandas `groupby(...).agg(agg_spec)` to compute
         numeric/feature aggregates.
      2. Independently compute provenance per group as:

            _prov_group = p1 + p2 + ... + pk

         where p1..pk are the `_prov` values of the rows in the group.
      3. Attach `_prov_group` as the `_prov` column in the aggregated
         dataframe.
    """
    if PROV_COLUMN not in df.columns:
        raise ValueError(f"Expected provenance column '{PROV_COLUMN}' in dataframe.")

    # Normalize `by` to a list for merging later
    if isinstance(by, str):
        group_keys = [by]
    else:
        group_keys = list(by)

    # Standard aggregation (excluding `_prov`)
    agg_df = (
        df.groupby(group_keys, dropna=False)
          .agg(agg_spec)
          .reset_index()
    )

    # Provenance aggregation
    prov_df = (
        df.groupby(group_keys, dropna=False)[PROV_COLUMN]
          .apply(_sum_provenances)
          .reset_index()
    )

    # Merge aggregated values and provenance
    result = agg_df.merge(prov_df, on=group_keys, how="inner")
    return result


#region Union (Set semantics)

def union_with_provenance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """
    For T₁ U T₂, provenance merges contributions from both sides.
    If a tuple (same *data* values, ignoring `_prov`) exists in one or
    both inputs, the result has a single row whose provenance is the
    semiring sum of all contributing provenances:

        _prov_result = p1 + p2 + ... (from both sides)

    We assume both dataframes share the same schema (same columns),
    including `_prov`.
    """
    if PROV_COLUMN not in df1.columns or PROV_COLUMN not in df2.columns:
        raise ValueError(f"Both inputs must contain a '{PROV_COLUMN}' column.")

    if set(df1.columns) != set(df2.columns):
        raise ValueError("Schemas of df1 and df2 must match for union.")

    non_prov_cols = [c for c in df1.columns if c != PROV_COLUMN]

    concatenated = pd.concat([df1, df2], ignore_index=True)

    prov_df = (
        concatenated
        .groupby(non_prov_cols, dropna=False)[PROV_COLUMN]
        .apply(_sum_provenances)
        .reset_index()
    )

    # Preserve column order as in df1
    return prov_df[non_prov_cols + [PROV_COLUMN]]


#region Intersection

def intersection_with_provenance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """
    A tuple is present in the result only if it appears in both T₁ and T₂
    (same data values, ignoring `_prov`). Its provenance is the *product*
    (semiring multiplication) of the provenance from each side, summed
    over all matching combinations:

        _prov_result = (p1 * q1) + (p2 * q2) + ...

    where p_i are provenance expressions from `df1` and q_j from `df2`
    for rows that share the same non-provenance values.
    """
    if PROV_COLUMN not in df1.columns or PROV_COLUMN not in df2.columns:
        raise ValueError(f"Both inputs must contain a '{PROV_COLUMN}' column.")

    if set(df1.columns) != set(df2.columns):
        raise ValueError("Schemas of df1 and df2 must match for intersection.")

    non_prov_cols = [c for c in df1.columns if c != PROV_COLUMN]

    # Join on all non-provenance columns
    merged = df1.merge(
        df2,
        on=non_prov_cols,
        how="inner",
        suffixes=("_left", "_right"),
    )

    if merged.empty:
        # Return an empty dataframe with the same schema as df1
        return df1.iloc[0:0].copy()

    # Combine provenance for each matching pair
    merged[PROV_COLUMN] = _product_provenances(
        merged[f"{PROV_COLUMN}_left"],
        merged[f"{PROV_COLUMN}_right"],
    )

    # Summarize multiple combinations per tuple by summing provenances
    prov_df = (
        merged
        .groupby(non_prov_cols, dropna=False)[PROV_COLUMN]
        .apply(_sum_provenances)
        .reset_index()
    )

    return prov_df[non_prov_cols + [PROV_COLUMN]]


# region Set Difference

def set_difference_with_provenance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """
    A tuple survives only if it exists in T₁ and *not* in T₂ (based on
    equality of all non-provenance columns). Its provenance is exactly
    T₁'s provenance; T₂ contributes nothing.
    """
    if PROV_COLUMN not in df1.columns or PROV_COLUMN not in df2.columns:
        raise ValueError(f"Both inputs must contain a '{PROV_COLUMN}' column.")

    if set(df1.columns) != set(df2.columns):
        raise ValueError("Schemas of df1 and df2 must match for set difference.")

    non_prov_cols = [c for c in df1.columns if c != PROV_COLUMN]

    # Use an anti-join: remove any row in df1 whose non-prov values exist in df2
    marker = "_merge_marker"
    tmp = df1.merge(
        df2[non_prov_cols].drop_duplicates(),
        on=non_prov_cols,
        how="left",
        indicator=marker,
    )

    result = tmp[tmp[marker] == "left_only"].drop(columns=[marker])

    # Preserve column order as in df1
    return result[df1.columns]


#region  Bag Difference (Multiset Difference)

def bag_difference_with_provenance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Like R8 but preserving multiplicities. We assume bag semantics are
    represented by *repeated rows* in the dataframes. For each tuple
    (same non-provenance values):

        multiplicity_result = max(0, multiplicity_T1 - multiplicity_T2)

    Provenance is inherited only from T₁: we keep a subset of T₁'s rows
    for each key according to the resulting multiplicity and drop the rest.
    T₂'s provenance never appears in the result.
    """
    if PROV_COLUMN not in df1.columns or PROV_COLUMN not in df2.columns:
        raise ValueError(f"Both inputs must contain a '{PROV_COLUMN}' column.")

    if set(df1.columns) != set(df2.columns):
        raise ValueError("Schemas of df1 and df2 must match for bag difference.")

    non_prov_cols = [c for c in df1.columns if c != PROV_COLUMN]

    # Count multiplicities in T₂ per non-provenance key
    counts2 = (
        df2
        .groupby(non_prov_cols, dropna=False)
        .size()
        .rename("count2")
        .reset_index()
    )

    # Attach a row-id so we can drop specific rows later
    df1_idx = df1.reset_index(drop=False).rename(columns={"index": "_row_id"})

    # Join df1 with counts from df2
    joined = df1_idx.merge(
        counts2,
        on=non_prov_cols,
        how="left",
    )
    joined["count2"] = joined["count2"].fillna(0).astype(int)

    # Within each key group in T₁, order rows and keep rows after `count2`
    joined["_rank"] = (
        joined
        .groupby(non_prov_cols, dropna=False)
        .cumcount()
        .astype(int)
    )

    # For a key with k rows in df1 and c rows in df2:
    #   keep rows where rank >= c  (0-based ranks)
    to_keep = joined[joined["_rank"] >= joined["count2"]]

    # Recover original column order (drop helper columns)
    result = to_keep[df1.columns.tolist()]

    return result
