import skrub
import cProfile, pstats

import pandas as pd

def main_merge():
    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})
    df2 = pd.DataFrame({"Country": ["Spain", "Belgium", "Italy"],
                        "Capital": ["Madrid", "Brussel", "Rome"]})
    
    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)

    
    joined_right = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="right",
        )
    )
    # print(joined_right)

def main_merge_print():
    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})
    df2 = pd.DataFrame({"Country": ["Spain", "Belgium", "Italy"],
                        "Capital": ["Madrid", "Brussel", "Rome"]})
    
    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)

    
    joined_right = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="right",
        )
    )
    print(joined_right.skb.eval())





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
    print("FIRST joined")
    print(joined_df)
    agg_df = joined_df.groupby("Country").agg({"population": 'sum'})
    print("SECOND aggregated")
    print(agg_df)
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)
    print("Third JOINED")
    print(agg_df.merge(people_table,on="Country", how="inner"))


def main():
    from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_dataop_callmethod, enter_provenance_mode_var
    from skrub._data_ops._evaluation import _Evaluator
    from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import set_provenance_evaluator, enter_provenance_mode_evaluator_eval_data_op, PROVENANCE_MODULE
    set_provenance_evaluator(
        _Evaluator,
        "_eval_data_op",
        enter_provenance_mode_evaluator_eval_data_op,
    )

    # Import the necessary libraries
    from skrub._data_ops._data_ops import CallMethod

    # Apply the monkey patch
    import functools

    # Save the original compute method so we can call it later
    # original_compute = CallMethod.compute

    # # Create the monkey-patched compute method
    # def patched_compute(self, e, mode, environment):
    #     # Handle preview mode specifically for `CallMethod`
    #     if mode == "preview":
    #         # Check if the method being called is 'agg' and inject `_prov`
    #         if self.method_name == "agg" and self.args:
    #             agg_args = self.args[0] if isinstance(self.args[0], dict) else {}
    #             if "_prov" not in agg_args:
    #                 PROVENANCE_MODULE.provenance_agg(self)
    #     return original_compute(self, e, mode, environment)

    # CallMethod.compute = patched_compute
    import skrub
    import pandas as pd
    # set_provenance(skrub._data_ops._data_ops.CallMethod,"compute", provenance_func=enter_provenance_mode_dataop_callmethod)
    set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)

    RUNS = 1
    from skrub._data_ops import _data_ops

    # takes a lot of time -> is not used -> disabled
    _data_ops._format_data_op_creation_stack = lambda: None

    # from skrub._data_ops import _evaluation

    # _original_evaluate = _evaluation.evaluate

    # def no_preview_evaluate(data_op, mode, environment, callbacks=None):
    #     if mode == "preview":
    #         return None
    #     return _original_evaluate(data_op, mode, environment, callbacks)

    # _evaluation.evaluate = no_preview_evaluate
    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})
    df2 = pd.DataFrame({"Country": ["Spain", "Belgium", "Italy"],
                        "Capital": ["Madrid", "Brussel", "Rome"]})
    
    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)

    
    


    # with cProfile.Profile() as profile:
    #     for _ in range(100-1):
    #         joined_right = (
    #             main_table
    #             .merge(
    #                 aux_table,
    #                 on="Country",
    #                 how="right",
    #             )
    #         )
    #     print(joined_right)

    import pandas as pd
    import numpy as np
    import skrub

    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})
    df2 = pd.DataFrame({"Country": ["Italy", "Belgium", "Italy"],
                        "City": ["Palermo", "Brussel", "Rome"],
                        'population':[1000,2000,1000],
                        })

    
    # main_table = skrub.var("main_table", df1)
    # aux_table = skrub.var("aux_table", df2)
    # people_table = skrub.var("people_table", df3)

    
    # print("FIRST joined")
    # print(joined_df)
    # agg_df = joined_df.groupby("Country").agg({"population": 'sum'})
    # print("SECOND aggregated")
    # print(agg_df)
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)
    # print("Third JOINED")
    # print(agg_df.merge(people_table,on="Country", how="inner"))

    with cProfile.Profile() as profile:
        for _ in range(100):
            main_table = skrub.var("main_table", df1)
            aux_table = skrub.var("aux_table", df2)
            people_table = skrub.var("people_table", df3)

            joined_df = (
                main_table
                .merge(
                    aux_table,
                    on="Country",
                    how="inner",
                )
            )
            agg_df = joined_df.groupby("Country").agg({"population": 'sum'})
            agg_df.merge(people_table,on="Country", how="inner")




    profiling_results = pstats.Stats(profile)
    profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

    profiling_results.print_stats(30)
    # 1.298 second with
    # 1.180 second without

    # _data_ops._format_data_op_creation_stack = lambda: None   -> reduces from 1.180 to 0.394   -> 640342 function calls (633848 primitive calls) in 0.394 seconds
    # with provenance enabled 759274 function calls (751379 primitive calls) in 0.495 seconds

# python -m pipelines.monkey_patching_v02.data_provenance.cprofiling_ASPJ

if __name__ == "__main__":
    main()