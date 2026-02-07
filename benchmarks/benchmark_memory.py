import numpy as np
import pandas as pd
import skrub

from monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance



def make_dataset(n_rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Main table
    df_main = pd.DataFrame({
        "user_id": rng.integers(0, n_rows // 10, size=n_rows),
        "category": rng.choice(["A", "B", "C", "D"], size=n_rows),
        "value": rng.normal(0, 1, size=n_rows),
        "text": rng.choice(
            [f"token_{i}" for i in range(100)],  # controls sparsity
            size=n_rows
        ),
    })

    # Lookup table for merge
    df_lookup = pd.DataFrame({
        "user_id": np.arange(n_rows // 10),
        "country": rng.choice(["FR", "DE", "US", "UK"], size=n_rows // 10),
        "segment": rng.choice(["S1", "S2", "S3"], size=n_rows // 10),
    })

    return df_main, df_lookup


import skrub
import logging

from time import perf_counter
from pathlib import Path
import csv
from skrub import TableVectorizer

# -------------------------
# Pipeline
# -------------------------
def run_pipeline(df_main, df_lookup, num_operations=1, verbose=False):
    # -------------------------
    # Initialization
    # -------------------------

    var_main = skrub.var("df_main", df_main)
    var_lookup = skrub.var("df_lookup", df_lookup)

    mem_with_initial = var_main.memory_usage(deep=True).sum().skb.preview()

    # -------------------------
    # Multiple Merges
    # -------------------------

    var_merged = var_main
    for i in range(num_operations):
        var_merged = var_merged.merge(var_lookup, on="user_id", how="left",  suffixes=('', f'_dup{i}'))
    mem_with_merge = var_merged.memory_usage(deep=True).sum().skb.preview()

    # -------------------------
    # Multiple Aggregations with skrub variables
    # -------------------------

    var_agg = var_main
    for i in range(num_operations):
        var_lookupi = skrub.var("var_lookup"+str(i), df_lookup)

        var_agg = var_agg.merge(
                var_lookupi,
                on="user_id",
                how="left",
            )
        
        var_agg = var_agg.groupby(["category", "user_id"], as_index=False).agg(
            country_count=("country", "count"),
        )

    # After the aggregation, get the memory usage
    mem_with_agg = var_agg.memory_usage(deep=True).sum().skb.preview()


    # -------------------------
    # Multiple TableVectorizer
    # -------------------------

    # here i need a vectorizer that can be applied multiple times one after another? maybe TableVectorizer is a good fit for that
    var_vectorized = var_main
    tv_with = TableVectorizer()
    for _ in range(num_operations):
        var_vectorized = var_vectorized.skb.apply(tv_with)

    mem_with_tv = var_vectorized.memory_usage(deep=True).sum().skb.preview()
    # -------------------------
    # Totals
    # -------------------------

    total_mem_with = (
        mem_with_initial
        + mem_with_merge
        + mem_with_agg
        + mem_with_tv
    )

    return {
        # initialization
        "mem_with_initialization": mem_with_initial,

        # merge
        "mem_with_merge": mem_with_merge,

        # aggregation
        "mem_with_aggregation": mem_with_agg,

        # tablevectorizer
        "mem_with_tablevectorizer": mem_with_tv,

        # totals
        "total_mem_with": total_mem_with,
    }


def memory_benchmark_for_scaling_operations(
        dataset_sizes=None,
        num_operations_list=[1, 2, 4, 8, 10],
        verbose=False,
        output_file_name= "memory_results_scaling.csv",
        enalbe_provenance=True,
        agg_func_for_prov_cols=list,
    ):
    if enalbe_provenance:
        enable_why_data_provenance(agg_func_over_prov_cols=agg_func_for_prov_cols)

    if dataset_sizes is None:
        dataset_sizes = [1_000]

    CSV_PATH = Path(f"{output_file_name}")

    FIELDNAMES = [
        "n_rows_input", "num_operations",
        "mem_with_initialization",

        "mem_with_merge",

        "mem_with_aggregation",

        "mem_with_tablevectorizer",

        "total_mem_with",
    ]

    write_header = not CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for n in dataset_sizes:
            for num_operations in num_operations_list:
                print(f"Running memory benchmark n={n}, operations={num_operations}")
                df_main, df_lookup = make_dataset(n)

                results = run_pipeline(
                    df_main,
                    df_lookup,
                    num_operations=num_operations,
                    verbose=verbose,
                )

                writer.writerow({
                    "n_rows_input": n,
                    "num_operations": num_operations,
                    **results,
                })

                print(f"Finished n={n}, operations={num_operations}")


if __name__ == "__main__":
    memory_benchmark_for_scaling_operations(enalbe_provenance=False, output_file_name="memory_benchmark_no_provenance.csv")
    memory_benchmark_for_scaling_operations(enalbe_provenance=True, agg_func_for_prov_cols=list, output_file_name="memory_benchmark_provenance_list.csv")
    memory_benchmark_for_scaling_operations(enalbe_provenance=True, agg_func_for_prov_cols=frozenset, output_file_name="memory_benchmark_provenance_frozenset.csv")
    memory_benchmark_for_scaling_operations(enalbe_provenance=True, agg_func_for_prov_cols="list_reduce", output_file_name="memory_benchmark_provenance_list_reduce.csv")
    memory_benchmark_for_scaling_operations(enalbe_provenance=True, agg_func_for_prov_cols="set_reduce", output_file_name="memory_benchmark_provenance_set_reduce.csv")
