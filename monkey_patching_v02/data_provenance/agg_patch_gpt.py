import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from functools import wraps

# --- Save originals ---
_original_df_agg = pd.DataFrame.aggregate
_original_gb_agg = DataFrameGroupBy.aggregate

# -----------------------------
# 1️⃣ Patched DataFrame.aggregate
# -----------------------------
@wraps(_original_df_agg)
def patched_df_agg(self, func=None, axis=0, *args, **kwargs):
    if axis != 0:
        return _original_df_agg(self, func=func, axis=axis, *args, **kwargs)

    prov_cols = [c for c in self.columns if c.startswith("_prov")]
    if not prov_cols:
        return _original_df_agg(self, func=func, axis=axis, *args, **kwargs)

    # List of columns that are not provenance columns
    data_cols = [c for c in self.columns if c not in prov_cols]

    # Run normal aggregation on data columns (no need to drop columns manually)
    base = _original_df_agg(self[data_cols], func=func, axis=axis, *args, **kwargs)

    # If the base is a Series, append provenance columns directly
    if isinstance(base, pd.Series):
        prov_rows = {c: self[c].tolist() for c in prov_cols}
        # Use `pd.concat()` on Series directly to minimize overhead
        return pd.concat([base, pd.Series(prov_rows)])

    # If the base is a DataFrame, handle provenance columns more efficiently
    # Instead of creating a DataFrame of lists and repeating, we just use vectorized operations
    prov_data = {c: self[c].tolist() for c in prov_cols}
    prov_df = pd.DataFrame(prov_data, index=base.index)
    
    # Directly concatenate base and provenance DataFrame
    return pd.concat([base, prov_df], axis=1)




# -----------------------------
# 2️⃣ Patched DataFrameGroupBy.aggregate
# -----------------------------
@wraps(_original_gb_agg)
def patched_gb_agg(self, func=None, *args, **kwargs):
    prov_cols = [c for c in self.obj.columns if c.startswith("_prov")]
    group_keys = list(self._grouper.names)

    # Columns to aggregate: exclude group keys and provenance
    agg_cols = [c for c in self.obj.columns if c not in group_keys + prov_cols]

    if not agg_cols and not prov_cols:
        return _original_gb_agg(self, func=func, *args, **kwargs)

    # --- Build function dict ---
    if isinstance(func, dict):
        func = func.copy()
        for c in prov_cols:
            if c not in func:
                func[c] = pd.Series.tolist
        return _original_gb_agg(self, func=func, *args, **kwargs)

    # Single function or list
    func_dict = {c: func for c in agg_cols}
    func_dict.update({c: pd.Series.tolist for c in prov_cols})

    return _original_gb_agg(self, func=func_dict, *args, **kwargs)

# -----------------------------
# 3️⃣ Apply the patch globally
# -----------------------------



def main_comparing_three_patching_functions():
    # pd.DataFrame.aggregate = patched_df_agg
    # pd.DataFrame.agg = patched_df_agg

    # RUNS = 1000

    import skrub._data_ops._data_ops as data_ops
    import skrub
    from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_var, enter_provenance_mode_dataop_callmethod
    set_provenance(data_ops.Var, "compute", provenance_func=enter_provenance_mode_var)
    # Monkey patch #1                                                                       without 2.033 -> with 2.4 -> only seen when .skb.eval() -> without .skb.eval() -> 1.7 seconds
    # provides provenance only when computing with skb.eval, not in the preview mode 
    # without .skb.eval does not interfere with preview processes

    # simple case
    # just execution, printing only once in the end
    # RUNS = 100
    # without patch with .skb.eval() ->     2751549 function calls (2736327 primitive calls) in 1.537 seconds
    # with patch with .skb.eval() ->        2839859 function calls (2821509 primitive calls) in 1.614 seconds
    # with patch without .skb.eval() ->     2220091 function calls (2216873 primitive calls) in 1.205 seconds
    # without patch without .skb.eval() ->  2217991 function calls (2214773 primitive calls) in 1.178 seconds

    # RUNS = 1000
    # without patch with .skb.eval() ->     27497949 function calls (27345927 primitive calls) in 15.277 seconds
    # with patch with .skb.eval() ->        28379159 function calls (28197009 primitive calls) in 15.903 seconds
    # with patch without .skb.eval() ->     22187491 function calls (22155473 primitive calls) in 11.838 seconds
    # without patch without .skb.eval() ->  22166491 function calls (22134473 primitive calls) in 11.751 seconds

    # complex case
    # just execution, printing only once in the end
    # RUNS = 100
    # without patch with .skb.eval() ->     19299470 function calls (19145671 primitive calls) in 10.401 seconds
    # with patch with .skb.eval() ->        19646300 function calls (19478671 primitive calls) in 10.674 seconds
    # with patch without .skb.eval() ->     16005243 function calls (15974346 primitive calls) in 8.402 seconds
    # without patch without .skb.eval() ->  15934212 function calls (15903317 primitive calls) in 8.381 seconds

    # RUNS = 1000
    # without patch with .skb.eval() ->     192980570 function calls (191443471 primitive calls) in 103.327 seconds     -> reran it 192980570 function calls (191443471 primitive calls) in 104.638 seconds
    # with patch with .skb.eval() ->        196446800 function calls (194772571 primitive calls) in 105.809 seconds     -> reran it 196446800 function calls (194772571 primitive calls) in 106.904 seconds
    # with patch without .skb.eval() ->     160042143 function calls (159734046 primitive calls) in 83.590 seconds
    # without patch without .skb.eval() ->  159332112 function calls (159024017 primitive calls) in 83.734 seconds

    # parameters to test
    # patch_compute = False

    # if patch_compute:
    #     set_provenance(skrub._data_ops._data_ops.CallMethod,"compute", provenance_func=enter_provenance_mode_dataop_callmethod)

    # skb_eval = True


    # Monkey patch #2   
    # patches pd.agg -> works in both preview and compute mode -> double overhead for .skb.eval()

    # simple case
    # just execution, printing only once in the end
    # RUNS = 100
    # without patch with .skb.eval() ->     2751549 function calls (2736327 primitive calls) in 1.516 seconds
    # with patch with .skb.eval() ->        2870496 function calls (2851546 primitive calls) in 1.637 seconds
    # with patch without .skb.eval() ->     2275738 function calls (2270592 primitive calls) in 1.233 seconds
    # without patch without .skb.eval() ->  2217991 function calls (2214773 primitive calls) in 1.190 seconds

    # RUNS = 1000
    # without patch with .skb.eval() ->     27497949 function calls (27345927 primitive calls) in 15.167 seconds
    # with patch with .skb.eval() ->        28685196 function calls (28497046 primitive calls) in 16.480 seconds    -> reran it 28685196 function calls (28497046 primitive calls) in 16.370 seconds
    # with patch without .skb.eval() ->     22741738 function calls (22691592 primitive calls) in 12.346 seconds
    # without patch without .skb.eval() ->  22166491 function calls (22134473 primitive calls) in 11.859 seconds


    # complex case
    # just execution, printing only once in the end
    RUNS = 100
    # without patch with .skb.eval() ->     19299470 function calls (19145671 primitive calls) in 10.484 seconds
    # with patch with .skb.eval() ->        20067479 function calls (19895152 primitive calls) in 11.002 seconds   -> reran it 19805879 function calls (19637552 primitive calls) in 10.854 seconds -> 19801479 function calls (19633952 primitive calls) in 10.890 seconds
    # with patch without .skb.eval() ->     16318321 function calls (16278098 primitive calls) in 8.722 seconds
    # without patch without .skb.eval() ->  15934212 function calls (15903317 primitive calls) in 8.424 seconds

    # RUNS = 1000
    # without patch with .skb.eval() ->     192980570 function calls (191443471 primitive calls) in 103.175 seconds
    # with patch with .skb.eval() ->        200658779 function calls (198937552 primitive calls) in 110.508 seconds
    # with patch without .skb.eval() ->     163171321 function calls (162771098 primitive calls) in 86.860 seconds
    # without patch without .skb.eval() ->  159332112 function calls (159024017 primitive calls) in 84.296 seconds

    # parameters to test
    patch_agg = True

    if patch_agg:
        pd.DataFrame.aggregate = patched_df_agg
        pd.DataFrame.agg = patched_df_agg
        DataFrameGroupBy.aggregate = patched_gb_agg
        DataFrameGroupBy.agg = patched_gb_agg

    skb_eval = True

    #                    0    _prov_x                                            _prov_y
    # population  4500  [1, 1, 3]  [281474976710656, 281474976710658, 28147497671...
    # City           3  [1, 1, 3]  [281474976710656, 281474976710658, 28147497671...

    # it should be like this
    #     population                                                 4500
    # City                                                          3
    # _prov_x                                               [1, 1, 3]
    # _prov_y       [281474976710656, 281474976710658, 28147497671...

    # Monkey patch #3   

    # simple case
    # just execution, printing only once in the end
    # RUNS = 100
    # without patch with .skb.eval() ->     2751549 function calls (2736327 primitive calls) in 1.547 seconds
    # with patch with .skb.eval() ->        2902300 function calls (2881750 primitive calls) in 1.656 seconds       -> reran it 2902300 function calls (2881750 primitive calls) in 1.639 seconds
    # with patch without .skb.eval() ->     2283242 function calls (2277696 primitive calls) in 1.249 seconds       -> reran it 2283242 function calls (2277696 primitive calls) in 1.242 seconds
    # without patch without .skb.eval() ->  2217991 function calls (2214773 primitive calls) in 1.183 seconds

    # RUNS = 1000
    # without patch with .skb.eval() ->     27497949 function calls (27345927 primitive calls) in 15.120 seconds
    # with patch with .skb.eval() ->        28685196 function calls (28497046 primitive calls) in 16.480 seconds    -> reran it 28685196 function calls (28497046 primitive calls) in 16.370 seconds
    # with patch without .skb.eval() ->     22816742 function calls (22762596 primitive calls) in 12.695 seconds
    # without patch without .skb.eval() ->  22166491 function calls (22134473 primitive calls) in 11.780 seconds

    # complex case
    # just execution, printing only once in the end
    # RUNS = 100
    # without patch with .skb.eval() ->     19299470 function calls (19145671 primitive calls) in 10.430 seconds
    # with patch with .skb.eval() ->        19813800 function calls (19640471 primitive calls) in 10.876 seconds
    # with patch without .skb.eval() ->     16170542 function calls (16133817 primitive calls) in 8.516 seconds
    # without patch without .skb.eval() ->  15934212 function calls (15903317 primitive calls) in 8.420 seconds

    # RUNS = 1000
    # without patch with .skb.eval() ->     192980570 function calls (191443471 primitive calls) in 103.363 seconds
    # with patch with .skb.eval() ->        198121800 function calls (196390571 primitive calls) in 107.381 seconds
    # with patch without .skb.eval() ->     161693342 function calls (161328117 primitive calls) in 85.224 seconds
    # without patch without .skb.eval() ->  159332112 function calls (159024017 primitive calls) in 83.824 seconds

    # parameters to test
    # patch_eval_data_op = True

    # if patch_eval_data_op:
    #     from skrub._data_ops._evaluation import _Evaluator
    #     from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import set_provenance_evaluator, enter_provenance_mode_evaluator_eval_data_op
    #     set_provenance_evaluator(
    #         _Evaluator,
    #         "_eval_data_op",
    #         enter_provenance_mode_evaluator_eval_data_op,
    #     )
    # skb_eval = True


    # main_merge_merge_agg2()
    df = pd.DataFrame({
        "A": [1, 2, 2],
        "B":[1,2,2]
    })

    df = skrub.var("df", df)

    import skrub
    import cProfile, pstats

    # simple case
    # if skb_eval:
    #     with cProfile.Profile() as profile:
    #         for _ in range(RUNS):
    #             df.groupby("A").agg({"B": "sum"}).skb.eval()
    # else:
    #     with cProfile.Profile() as profile:
    #         for _ in range(RUNS):
    #             df.groupby("A").agg({"B": "sum"})

    # complex case
    if skb_eval:
        with cProfile.Profile() as profile:
            for _ in range(RUNS):
                main_merge_isin_agg_simpler_eval()
                main_inner_merge_groupby_agg_merge_eval()

    else:
        with cProfile.Profile() as profile:
            for _ in range(RUNS):
                main_merge_isin_agg_simpler()
                main_inner_merge_groupby_agg_merge()


    profiling_results = pstats.Stats(profile)
    profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

    profiling_results.print_stats(30)
    if skb_eval:
        print(main_merge_isin_agg_simpler_eval())
        print(main_inner_merge_groupby_agg_merge_eval())
    else:
        print(main_merge_isin_agg_simpler())
        print(main_inner_merge_groupby_agg_merge())
    
    # # df = pd.DataFrame({
    # "A": [1, 2, 2],
    # "_prov_A": ["x", "y", "z"],
    # "_prov_B": ["p", "q", "r"],
    # })

    # print(df.agg(["sum", "mean"]))


def main_inner_merge_groupby_agg_merge():
    import pandas as pd
    import numpy as np
    import skrub

    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})
    df2 = pd.DataFrame({"Country": ["Italy", "Belgium", "Italy"],
                        "City": ["Palermo", "Brussel", "Rome"],
                        'population':[1000,2000,1000],
                        })

    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)


    joined_df = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="inner",
        )
    )
    # print(joined_df)
    agg_df = joined_df.groupby("Country").agg({"population": 'sum'})
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    # print(agg_df.merge(people_table,on="Country", how="inner"))
    return agg_df.merge(people_table,on="Country", how="inner")

def main_inner_merge_groupby_agg_merge_eval():
    import pandas as pd
    import numpy as np
    import skrub

    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})
    df2 = pd.DataFrame({"Country": ["Italy", "Belgium", "Italy"],
                        "City": ["Palermo", "Brussel", "Rome"],
                        'population':[1000,2000,1000],
                        })

    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)


    joined_df = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="inner",
        )
    )
    # print(joined_df)
    agg_df = joined_df.groupby("Country").agg({"population": 'sum'})
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    # print(agg_df.merge(people_table,on="Country", how="inner"))
    return agg_df.merge(people_table,on="Country", how="inner").skb.eval()

def main_merge_isin_agg_simpler():
    import pandas as pd
    import numpy as np
    import skrub

    df1 = pd.DataFrame({
        "Country": ["USA", "Italy", "Georgia", "Belgium"]
    })

    df2 = pd.DataFrame({
        "Country": ["Italy", "Belgium", "Italy", "USA"],
        "City": ["Palermo", "Brussels", "Rome", "NYC"],
        "population": [1000, 2000, 1500, 3000],
    })

    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)

    joined = main_table.merge(aux_table, on="Country", how="inner")
    # print(joined)

    # ---- isin filtering stage ----
    european_countries = ["Italy", "Belgium"]
    filtered = joined[joined["Country"].isin(european_countries)]

    # print("filtered")
    # print(filtered)

    # ---- aggregation without creating new column names ----
    # TODO: if someone grabs any column -> send prov columns with it!
    # result = filtered[["population", "City"]].agg({
    #     "population": "sum",   # sum the population column
    #     "City": "nunique"      # count unique cities
    # })
    result = filtered.agg({
        "population": "sum",   # sum the population column
        "City": "nunique"      # count unique cities
    })
    return result

def main_merge_isin_agg_simpler_eval():
    import pandas as pd
    import numpy as np
    import skrub

    df1 = pd.DataFrame({
        "Country": ["USA", "Italy", "Georgia", "Belgium"]
    })

    df2 = pd.DataFrame({
        "Country": ["Italy", "Belgium", "Italy", "USA"],
        "City": ["Palermo", "Brussels", "Rome", "NYC"],
        "population": [1000, 2000, 1500, 3000],
    })

    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)

    joined = main_table.merge(aux_table, on="Country", how="inner")
    # print(joined)

    # ---- isin filtering stage ----
    european_countries = ["Italy", "Belgium"]
    filtered = joined[joined["Country"].isin(european_countries)]

    # print("filtered")
    # print(filtered)

    # ---- aggregation without creating new column names ----
    # TODO: if someone grabs any column -> send prov columns with it!
    # result = filtered[["population", "City"]].agg({
    #     "population": "sum",   # sum the population column
    #     "City": "nunique"      # count unique cities
    # })
    result = filtered.agg({
        "population": "sum",   # sum the population column
        "City": "nunique"      # count unique cities
    })
    return result.skb.eval()

def main_validation_agg_patch():
    DataFrameGroupBy.aggregate = patched_gb_agg
    DataFrameGroupBy.agg = patched_gb_agg
    pd.DataFrame.aggregate = patched_df_agg
    pd.DataFrame.agg = patched_df_agg

    import skrub._data_ops._data_ops as data_ops
    import skrub
    from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_var, enter_provenance_mode_dataop_callmethod
    set_provenance(data_ops.Var, "compute", provenance_func=enter_provenance_mode_var)


    # main_inner_merge_groupby_agg_merge()
    # main_merge_isin_agg_simpler()


def main():
    main_comparing_three_patching_functions()

if __name__ == "__main__":
    main()


# def main():
#     pd.DataFrame.aggregate = patched_df_agg
#     pd.DataFrame.agg = patched_df_agg
#     DataFrameGroupBy.aggregate = patched_gb_agg
#     DataFrameGroupBy.agg = patched_gb_agg

#     df = pd.DataFrame({
#     "A": [1, 2, 2],
#     "B": [1, 2, 2],
#     "_prov_A": [0, 1, 2],
#     "_prov_B": [10, 11, 12]
#     })

#     print(df.groupby("A").agg("sum"))


# if __name__ == "__main__":
#     main()