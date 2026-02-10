from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from benchmarks.benchmark_utils import make_dataset
from skrub import TableVectorizer
from time import perf_counter
from skrub import ApplyToCols
from pathlib import Path
import skrub
import csv


# python -m monkey_patching_v02.data_provenance.benchmarking_time_mpv02
# -------------------------
# Pipeline
# -------------------------
def run_pipeline(df_main, df_lookup, verbose=False):
    # --- Initialization time for one dataframe ---
    t0 = perf_counter()
    var_main = skrub.var("df_main", df_main)
    initialization_time = perf_counter() - t0

    var_lookup = skrub.var("df_lookup",df_lookup)

    if verbose:
        print("df_main")
        print(var_main)
        print("var_lookup")
        print(var_lookup)

    
    # --- Merge ---
    t0 = perf_counter()
    var_merged = var_main.merge(var_lookup, on="user_id", how="left")
    merge_time = perf_counter() - t0

    if verbose:
        print("var_merged")
        print(var_merged)


    # --- Aggregation ---
    t0 = perf_counter()
    var_agg = (
        var_merged
        .groupby(["category"], as_index=False)
        .agg({
            "text": "count",
        })
    )
    aggregation_time = perf_counter() - t0

    if verbose:
        print("var_agg")
        print(var_agg)

    # var_agg does not contain many columns
    # Applying three different scikit-learn estimators to var_merged
    # --- TableVectorizer ---
    t0 = perf_counter()
    tv = TableVectorizer()
    X = var_merged.skb.apply(tv)
    tablevectorizer_time = perf_counter() - t0
    if verbose:
        print("df_tableVectorized")
        print(X)


    ohe = ApplyToCols(OneHotEncoder(sparse_output=False), cols= ["category"])
    # --- OneHotEncoder ---
    t0 = perf_counter()
    X = var_merged.skb.apply(ohe)
    oneHotEncoder_time = perf_counter() - t0
    if verbose:
        print("df_OneHotEncoded")
        print(X)

    standard_scaler = ApplyToCols(StandardScaler(), cols=["value"])
    # --- StandardScaler ---
    t0 = perf_counter()
    X = var_merged.skb.apply(standard_scaler)
    standardScaler_time = perf_counter() - t0
    if verbose:
        print("df_StandardScaled")
        print(X)

    
    

    total_time = initialization_time + merge_time + aggregation_time + tablevectorizer_time
    # total_time = merge_time
    return {
        "initialization_time":initialization_time,
        "merge_time": merge_time,
        "aggregation_time": aggregation_time,
        "tablevectorizer_time": tablevectorizer_time,
        "total_time": total_time,
        "oneHotEncoder_time":oneHotEncoder_time,
        "standardScaler_time":standardScaler_time,
        "n_features": X.skb.eval().shape[1],
    }



def benchmark(enable_provenance=False,  n_runs=5, dataset_sizes=None, verbose=False,
              output_file_name= "memory_results_scaling.csv",
              agg_func_for_prov_cols=list,):
    if dataset_sizes is None:
        dataset_sizes = [1000, 10_000, 100_000, 1_000_000, 10_000_000]
        # dataset_sizes = [1_000_000_000]

    CSV_PATH = Path(f"benchmark_logs/{output_file_name}")

    FIELDNAMES = [
        "initialization_time",
        "n_rows_input",
        "merge_time",
        "aggregation_time",
        "tablevectorizer_time",
        "oneHotEncoder_time",
        "standardScaler_time",
        "total_time",
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

            for _ in range(n_runs):
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
    benchmark(enable_provenance=False, n_runs=5, output_file_name="benchmark_runtime_n_rows_no_provenance.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols=list, output_file_name="benchmark_runtime_n_rows_provenance_list.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols=frozenset, output_file_name="benchmark_runtime_n_rows_provenance_frozenset.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols="list_reduce", output_file_name="benchmark_runtime_n_rows_provenance_list_reduce.csv")
    benchmark(enable_provenance=True, n_runs=5, agg_func_for_prov_cols="set_reduce", output_file_name="benchmark_runtime_n_rows_provenance_set_reduce.csv")