from __future__ import annotations

import pandas as pd

from skrub.datasets import fetch_employee_salaries

from rdepro_skrub_provenance.provenance_utils import (
    with_provenance,
    project_with_provenance,
    select_with_provenance,
    groupby_aggregate_with_provenance,
    union_with_provenance,
)
from rdepro_skrub_provenance.simple_ops import rename_column_with_prov


def load_skrub_employee_salaries() -> pd.DataFrame:
    """
    Load the 'employee_salaries' dataset from Skrub and return the main table X.

    Skrub returns a Bunch containing:
        - X: features as a DataFrame
        - y: target as a DataFrame or Series
    """
    dataset = fetch_employee_salaries()
    X, y = dataset.X, dataset.y
    return X


def demo_provenance_pipeline():
    """
    Demonstrate a small provenance-aware pipeline applied to the Skrub
    'employee_salaries' dataset.

    Steps:
      - R0: initialize provenance with with_provenance()
      - R1: rename a column (rename_column_with_prov)
      - R2: project on a subset of columns (project_with_provenance)
      - R3: filter rows (select_with_provenance)
      - R5: groupby aggregation (groupby_aggregate_with_provenance)
      - R6: union two subsets (union_with_provenance)
    """

    # 1) Load the dataset
    employees = load_skrub_employee_salaries()
    print("Original employees shape:", employees.shape)
    print(employees.head())

    # 2) R0 – Initialize provenance (attach a leaf provenance tag)
    employees_prov = with_provenance(employees, source_name="employee_salaries_X")
    assert "_prov" in employees_prov.columns
    print("\nAfter with_provenance:")
    print(employees_prov[["_prov"]].head())

    # 3) R1 – Rename a column, e.g., 'department' -> 'dept'
    if "department" in employees_prov.columns:
        renamed = rename_column_with_prov(employees_prov, "department", "dept")

        # Check: provenance column must remain unchanged
        assert renamed["_prov"].equals(employees_prov["_prov"])

        print("\nAfter rename_column_with_prov('department' -> 'dept'):")
        print(renamed[["dept", "_prov"]].head())
    else:
        renamed = employees_prov
        print("\nColumn 'department' not found, skipping renaming step.")

    # 4) R2 – Projection: keep a subset of columns + provenance
    cols_to_keep = []
    for c in ["dept", "department", "gender", "salary_or_hourly", "annual_salary"]:
        if c in renamed.columns:
            cols_to_keep.append(c)

    proj = project_with_provenance(renamed, columns=cols_to_keep)
    print("\nAfter project_with_provenance:")
    print(proj.head())

    # 5) R3 – Selection: keep only rows matching a condition
    # Example: keep rows where gender == 'F'
    if "gender" in proj.columns:
        mask_female = proj["gender"] == "F"
        selected = select_with_provenance(proj, mask=mask_female)

        # Check: provenance values should be a subset of original ones
        assert set(selected["_prov"]).issubset(set(proj["_prov"]))

        print("\nAfter select_with_provenance (gender == 'F'):")
        print(selected.head())
    else:
        selected = proj
        print("\nColumn 'gender' not found, skipping selection step.")

    # 6) R5 – Aggregation: count by department and gender
    groupby_cols = []
    for c in ["dept", "department"]:
        if c in selected.columns:
            groupby_cols.append(c)
            break  # only keep the first matching name

    if "gender" in selected.columns and groupby_cols:
        agg_spec = {"gender": "count"}

        grouped = groupby_aggregate_with_provenance(
            selected,
            by=groupby_cols,
            agg_spec=agg_spec,
        )

        print("\nAfter groupby_aggregate_with_provenance:")
        print(grouped.head())
        print("\nExample provenance expression after aggregation:")
        print(grouped[["_prov"]].head())
    else:
        grouped = selected
        print("\nMissing 'gender' or department column, skipping aggregation (R5).")

    # 7) R6 – Union: create two artificial subsets and merge them
    n = len(selected)
    if n > 0:
        left_part = selected.iloc[: n // 2].copy()
        right_part = selected.iloc[n // 3 :].copy()  # partial overlap

        union_df = union_with_provenance(left_part, right_part)

        print("\nAfter union_with_provenance(left_part, right_part):")
        print(union_df.head())

        print("\nExample provenance for a unioned row:")
        print(union_df["_prov"].head())
    else:
        print("\nSelected dataframe is empty, skipping union test.")


if __name__ == "__main__":
    demo_provenance_pipeline()

