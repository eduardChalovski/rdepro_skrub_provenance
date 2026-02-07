import numpy as np
import pandas as pd

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier, plot_importance      # i thought xgboost is defined already in scikit learn
import shap
import warnings
import skrub
from skrub import ToDatetime
from skrub import DatetimeEncoder
import cProfile, pstats


def with_provenance_integers_shifted_simplified_for_profiling(df: pd.DataFrame, table_id: str) -> pd.DataFrame:
    df = df.copy()
    row_ids = np.arange(len(df), dtype=np.int64)            # TODO: consider using indices instead of len(df)
    df["_prov" + str(table_id)] = (np.int64(table_id) << 48) | row_ids      # pack_prov(table_id, row_id)
    return df

def main():
    print("Profiling of agg: Started")
    df_main, df_lookup = make_dataset(1_000_000,0)
    df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
    df_merged = df_main.merge(df_lookup, on="user_id", how="left")


    with cProfile.Profile() as profile:
        for _ in range(10):
            df_agg = (
                    df_merged
                    .groupby(["category"], as_index=False)
                    .agg(
                        text=("text","count"),
                        _prov0=("_prov0",list),
                        _prov1=("_prov1",list),
                    )
                )

    profiling_results = pstats.Stats(profile)
    profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

    profiling_results.print_stats(30)
    print((
                    df_merged
                    .groupby(["category"], as_index=False)
                    .agg(
                        text=("text","count"),
                        _prov0=("_prov0",list),
                        _prov1=("_prov1",list),
                    )
                ))

if __name__ == "__main__":
    main()

# python .\monkey_patching_v02\data_provenance\testing_another_agg_performance.py

# 57918 function calls (56298 primitive calls) in 1.747 seconds
# def main():
#     print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(1_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")
#     df_groupby = df_merged.groupby(["category"], as_index=False)

#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_groupby
#                     .agg({
#                         "text": "count",
                        
#                     })
#                 )
            
                        
#             df_agg2 = (
#                     df_groupby
#                     .agg({
#                         "_prov0":list,
#                     })
#                 )
#             df_agg3 = (
#                     df_groupby
#                     .agg({
#                         "_prov1":list,
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((df_groupby
#                     .agg({
#                         "text": "count",
#                     })
#                 ))
#     print(df_groupby.agg({
#                         "_prov0":list,
#                     }))
#     print(df_groupby.agg({
#                         "_prov1":list,
#                     }))

# 44088 function calls (42738 primitive calls) in 1.727 seconds
# prov_cols are not yet concatanated with the result_df -> + overhead
# def main():
#     print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(1_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")
#     df_groupby = df_merged.groupby(["category"], as_index=False)

#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_groupby
#                     .agg({
#                         "text": "count",
                        
#                     })
#                 )
            
                        
#             df_agg2 = (
#                     df_groupby
#                     .agg({
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((df_groupby
#                     .agg({
#                         "text": "count",
#                     })
#                 ))
#     print(df_groupby.agg({
#                         "_prov0":list,
#                         "_prov1":list,
#                     }))

# if __name__ == "__main__":
#     main() 


# 30488 function calls (29408 primitive calls) in 1.711 seconds

# def main():
#     print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(1_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")
#     df_groupby = df_merged.groupby(["category"], as_index=False)

#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_groupby
#                     .agg({
#                         "text": "count",
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 ))

# if __name__ == "__main__":
#     main()

# 20203 function calls (19791 primitive calls) in 0.525 seconds

# print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(1_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")


#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                     })
#                 ))


# 33157 function calls (32207 primitive calls) in 3.044 seconds

# def main():
#     print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(1_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")


#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":set,
#                         "_prov1":set,
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":set,
#                         "_prov1":set,
#                     })
#                 ))


# 33757 function calls (32647 primitive calls) in 2.027 seconds

# print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(1_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")


#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 ))




# 20203 function calls (19791 primitive calls) in 5.010 seconds
# print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(10_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")


#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                     })
#                 ))


# 33757 function calls (32647 primitive calls) in 20.278 seconds
# print("Profiling of agg: Started")
#     df_main, df_lookup = make_dataset(10_000_000,0)
#     df_main, df_lookup = with_provenance_integers_shifted_simplified_for_profiling(df_main,0), with_provenance_integers_shifted_simplified_for_profiling(df_lookup, 1)
#     df_merged = df_main.merge(df_lookup, on="user_id", how="left")


#     with cProfile.Profile() as profile:
#         for _ in range(10):
#             df_agg = (
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 )

#     profiling_results = pstats.Stats(profile)
#     profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

#     profiling_results.print_stats(30)
#     print((
#                     df_merged
#                     .groupby(["category"], as_index=False)
#                     .agg({
#                         "text": "count",
#                         "_prov0":list,
#                         "_prov1":list,
#                     })
#                 ))