from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
from benchmarks.benchmark_utils import make_dataset
from time import perf_counter
from pathlib import Path
import skrub
import csv


# -------------------------
# Pipeline
# -------------------------
def run_pipeline(df_main, df_lookup, num_operations=1):
    # -------------------------
    # Initialization
    # -------------------------
    var_main, var_lookup = skrub.var("df_main", df_main), skrub.var("df_lookup", df_lookup)

    # -------------------------
    # Multiple Merges
    # -------------------------
    var_merged = var_main
    t0 = perf_counter()
    for i in range(num_operations):
        var_merged = var_merged.merge(var_lookup, on="user_id", how="left",  suffixes=('', f'_dup{i}'))
    merge_time = perf_counter() - t0

    # -------------------------
    # Multiple Aggregations with skrub variables
    # -------------------------
    var_agg = var_main
    t0 = perf_counter()

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

    # After the aggregation, get the runtime overhead
    merge_agg_time = perf_counter() - t0



    return {
        # merge
        "runtime_merge": merge_time,

        # aggregation
        "runtime_aggregation": merge_agg_time,

    }


def benchmark_runtime_n_operators(
        dataset_sizes=None,
        num_operations_list=[1, 2, 4, 6, 8, 10],
        output_file_name= "runtime_results_scaling.csv",
        enalbe_provenance=True,
        agg_func_for_prov_cols=list,
    ):
    if enalbe_provenance:
        enable_why_data_provenance(agg_func_over_prov_cols=agg_func_for_prov_cols)

    if dataset_sizes is None:
        dataset_sizes = [100_000]

    CSV_PATH = Path(f"benchmark_logs/{output_file_name}")

    FIELDNAMES = [
        "n_rows_input", "num_operations",
        "runtime_merge",
        "runtime_aggregation",

    ]

    write_header = not CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for n in dataset_sizes:
            for num_operations in num_operations_list:
                print(f"Running runtime benchmark n={n}, operations={num_operations}")
                df_main, df_lookup = make_dataset(n)

                results = run_pipeline(
                    df_main,
                    df_lookup,
                    num_operations=num_operations,
                )

                writer.writerow({
                    "n_rows_input": n,
                    "num_operations": num_operations,
                    **results,
                })

                print(f"Finished n={n}, operations={num_operations}")


if __name__ == "__main__":
    # benchmark_runtime_n_operators(enalbe_provenance=False, output_file_name="benchmark_runtime_n_operators_no_provenance.csv")
    # benchmark_runtime_n_operators(enalbe_provenance=True, agg_func_for_prov_cols=list, output_file_name="benchmark_runtime_n_operators_provenance_list.csv")
    # benchmark_runtime_n_operators(enalbe_provenance=True, agg_func_for_prov_cols=frozenset, output_file_name="benchmark_runtime_n_operators_provenance_frozenset.csv")
    benchmark_runtime_n_operators(enalbe_provenance=True, agg_func_for_prov_cols="list_reduce", output_file_name="benchmark_runtime_n_operators_provenance_list_reduce.csv")
    # benchmark_runtime_n_operators(enalbe_provenance=True, agg_func_for_prov_cols="set_reduce", output_file_name="benchmark_runtime_n_operators_provenance_set_reduce.csv")
