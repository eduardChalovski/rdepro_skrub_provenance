from monkey_patching_v02.monkey_patching_v0_2_utilizing_jeannes_prov_utils import enter_provenance_mode as provenance_func_name_v0_2
from monkey_patching_v02.monkey_patching_v0_2_utilizing_jeannes_prov_utils import set_provenance
import skrub
import pandas as pd



#from monkey_patching_v0 import enter_provenance_mode
#set_provenance(skrub._data_ops._data_ops.DataOp, "__call__")


def main():
    pd.options.display.width = 1200
    pd.options.display.max_colwidth = 1000
    pd.options.display.max_columns = 100
    set_provenance(skrub._data_ops._data_ops.DataOp, "__call__",provenance_func_name_v0_2)

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
    joined_outer = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="outer",
        )
    )
    
    joined = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="left",
        )
    )

    
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined
        .merge(
            people_table,
            on="Country",
            how="left",
        )
    )
    """         left_on and right_on are not yet supproted -> breaks the code
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Home Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined
        .merge(
            people_table,
            left_on="Country", right_on="Home Country",
            how="left",
        )
    )"""

    joined2_aggregated = joined2.groupby("Country").agg("count")
    
    joined2_aggregated.merge(joined_outer, on="Country", how="right")
    """ SHOWCASE OF A SIMPLE AGGREGATION
    # region Aggregation
    # simplify the task -> no nesting -> create a df for that
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Home Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    simple_grouped_by_df = people_table.groupby("Home Country").agg("count")
    """
    # region TASK:






    # region make it nested
    
    

if __name__ == "__main__":
    main()

# python -m monkey_patching_v02.easy_ASPJ_pipeline_search_for_better_provenance