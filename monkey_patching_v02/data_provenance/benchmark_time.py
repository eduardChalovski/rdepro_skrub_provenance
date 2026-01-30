import numpy as np
import pandas as pd
import skrub


# python -m monkey_patching_v02.data_provenance.benchmarking_time_mpv02
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
import logging
from skrub import TableVectorizer

# -------------------------
# Pipeline
# -------------------------
def run_pipeline(df_main, df_lookup, verbose=False):
    # --- Initialization ---
    t0 = perf_counter()
    df_main, df_lookup = skrub.var("df_main", df_main), skrub.var("df_lookup",df_lookup)
    initialization_time = perf_counter() - t0
    if verbose:
        print("df_main")
        print(df_main)
        print("df_lookup")
        print(df_lookup)

    
    # --- Merge ---
    t0 = perf_counter()
    df_merged = df_main.merge(df_lookup, on="user_id", how="left")
    merge_time = perf_counter() - t0

    if verbose:
        print("df_merged")
        print(df_merged)


    # --- Aggregation ---
    t0 = perf_counter()
    df_agg = (
        df_merged
        .groupby(["category"], as_index=False)
        .agg({
            "text": "count",
        })
    )
    aggregation_time = perf_counter() - t0

    if verbose:
        print("df_agg")
        print(df_agg)

    # --- TableVectorizer ---
    t0 = perf_counter()
    tv = TableVectorizer()
    X = df_merged.skb.apply(tv)
    tablevectorizer_time = perf_counter() - t0
    if verbose:
        print("df_tableVectorized")
        print(X)

    total_time = initialization_time + merge_time + aggregation_time + tablevectorizer_time
    # total_time = merge_time
    return {
        "initialization_time":initialization_time,
        "merge_time": merge_time,
        "aggregation_time": aggregation_time,
        "tablevectorizer_time": tablevectorizer_time,
        "total_time": total_time,
        "rows_after_merge": len(df_merged.skb.eval()),
        "rows_after_aggregation": len(df_agg.skb.eval()),
        "n_features": X.skb.eval().shape[1],
    }



def benchmark(enable_provenance=False,  n_runs=5, dataset_sizes=None, verbose=False,
              output_file_name= "memory_results_scaling.csv",
              agg_func_for_prov_cols=list,):
    if dataset_sizes is None:
        dataset_sizes = [1000, 10_000, 100_000, 1_000_000, 10_000_000]
        # dataset_sizes = [1_000_000_000]

    CSV_PATH = Path(f"{output_file_name}")

    FIELDNAMES = [
        "initialization_time",
        "n_rows_input",
        "merge_time",
        "aggregation_time",
        "tablevectorizer_time",
        "total_time",
        "rows_after_merge",
        "rows_after_aggregation",
        "n_features",
    ]

    if enable_provenance:
        enable_why_data_provenance(agg_func_over_prov_cols=agg_func_for_prov_cols)



    write_header = not CSV_PATH.exists()

    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if write_header:
            writer.writeheader()

        for n in dataset_sizes:
            print(f"Running n={n}")

            df_main, df_lookup = make_dataset(n)
            # accumulator
            sums = {k: 0.0 for k in FIELDNAMES if k != "n_rows_input"}

            for run in range(n_runs):
                timings = run_pipeline(df_main, df_lookup, verbose)

                for k in sums:
                    sums[k] += timings[k]

            # compute averages
            averages = {k: sums[k] / n_runs for k in sums}

            row = {
                "n_rows_input": n,
                **averages,
            }

            writer.writerow(row)
            print(f"Finished n={n} (averaged over {n_runs} runs)")

if __name__ == "__main__":
    # benchmark(provenance_enable_why_data=False, n_runs=50)
    # benchmark(provenance_enable_why_data=True, n_runs=50)
    benchmark(enable_provenance=False, n_runs=5, output_file_name="speed_benchmark_no_provenance.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols=list, output_file_name="speed_benchmark_provenance_list.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols=frozenset, output_file_name="speed_benchmark_provenance_frozenset.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols="list_reduce", output_file_name="speed_benchmark_provenance_list_reduce.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols="set_reduce", output_file_name="speed_benchmark_provenance_set_reduce.csv")
   

# python -m monkey_patching_v02.data_provenance.benchmarking_time_mpv02