# from monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_dataop
import skrub
import pandas as pd

# python -m monkey_patching_v02.data_provenance.testing_integer_ids_provenance

from monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import evaluate_provenance

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
    print(joined_right)
    print("#"*80)
    print("evaluate_provenance(joined_right)")
    print(evaluate_provenance(joined_right.skb.preview()))
    print()
    print("now by passing a dataop")
    print()
    print(evaluate_provenance(joined_right))

def main_agg():
    import pandas as pd
    import numpy as np
    import skrub
    
    df = pd.DataFrame({
        'Country': ['Italy', 'Belgium', 'Italy'],
        'Capital': ['Rome', 'Brussel', 'Rome'],
        'population':[1000,2000,1000],
    })
    df_var = skrub.var("main_table", df)
    result_agg = df_var.agg({"population": 'sum'})
    print(result_agg)

def main_left_merge_agg():
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


    joined_right = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="left",
        )
    )
    print(joined_right)
    print(joined_right.agg({"population": 'sum'}))


def main_inner_merge_agg():
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


    joined_right = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="inner",
        )
    )
    print(joined_right)
    print(joined_right.agg({"population": 'sum'}))


def main_left_merge_groupby_agg():
    import pandas as pd
    import numpy as np
    import skrub

    df1 = pd.DataFrame({"Country": ["USA", "Italy", "Belgium"]})
    df2 = pd.DataFrame({"Country": ["Italy", "Belgium", "Italy"],
                        "City": ["Palermo", "Brussel", "Rome"],
                        'population':[1000,2000,1000],
                        })

    main_table = skrub.var("main_table", df1)
    aux_table = skrub.var("aux_table", df2)


    joined_right = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="left",
        )
    )
    print(joined_right)
    print(joined_right.groupby("Country").agg({"population": 'sum'}))


def main_inner_merge_groupby_agg():
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


    joined_right = (
        main_table
        .merge(
            aux_table,
            on="Country",
            how="inner",
        )
    )
    print(joined_right)
    print(joined_right.groupby("Country").agg({"population": 'sum'}))


def main_inner_merge_groupby_agg_eval():
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
    print(joined_df)
    print(joined_df.groupby("Country").agg({"population": 'sum'}).skb.eval())

def main_merge_merge():
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

    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined_df
        .merge(
            people_table,
            on="Country",
            how="left",
        )
    )

    print(joined2)
    

def main_merge_merge_agg():
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

    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined_df
        .merge(
            people_table,
            on="Country",
            how="left",
        )
    )

    print(joined2.agg("count"))

def main_merge_merge_agg2():
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

    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined_df
        .merge(
            people_table,
            on="Country",
            how="left",
        )
    )

    print(joined2.agg("size"))


def main_merge_merge_agg3():
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

    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined_df
        .merge(
            people_table,
            on="Country",
            how="left",
        )
    )

    print(joined2.agg(pd.Series.count))

def main_merge_merge_agg4():
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

    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    joined2 = (
        joined_df
        .merge(
            people_table,
            on="Country",
            how="left",
        )
    )

    print(joined2.agg(lambda s: s.size))

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
    print(joined_df)
    agg_df = joined_df.groupby("Country").agg({"population": 'sum'})
    df3 = pd.DataFrame({"Name": ["Person1", "Person2", "Person3"],
                    "Country": ["Italy", "Italy", "Germany"]})#.set_index("Country")
    people_table = skrub.var("people_table", df3)

    print(agg_df.merge(people_table,on="Country", how="inner"))


def main_merge_tablevectorizer():
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
    print(joined_df)
    print(joined_df.skb.apply(skrub.TableVectorizer()))

def main_dataops_intro_preview():
    from skrub.datasets import fetch_employee_salaries

    training_data = fetch_employee_salaries(split="train").employee_salaries
    data_var = skrub.var("data", training_data)
    X = data_var.drop("current_annual_salary", axis=1).skb.mark_as_X()
    y = data_var["current_annual_salary"].skb.mark_as_y()
    from skrub import TableVectorizer

    vectorizer = TableVectorizer()

    X_vec = X.skb.apply(vectorizer)
    from sklearn.ensemble import HistGradientBoostingRegressor

    hgb = HistGradientBoostingRegressor()

    predictor = X_vec.skb.apply(hgb, y=y)#.skb.eval()
    print(predictor)

def main_dataops_intro_eval():
    from skrub.datasets import fetch_employee_salaries

    training_data = fetch_employee_salaries(split="train").employee_salaries
    data_var = skrub.var("data", training_data)
    X = data_var.drop("current_annual_salary", axis=1).skb.mark_as_X()
    y = data_var["current_annual_salary"].skb.mark_as_y()
    from skrub import TableVectorizer

    vectorizer = TableVectorizer()

    X_vec = X.skb.apply(vectorizer)
    from sklearn.ensemble import HistGradientBoostingRegressor

    hgb = HistGradientBoostingRegressor()

    predictor = X_vec.skb.apply(hgb, y=y).skb.eval()
    print(predictor)

def main_scaler():
    from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, MinMaxScaler

    from skrub import SquashingScaler

def main_selectors():
    skrub.selectors._base.All


def main_PCA():
    from sklearn.decomposition import PCA

def main_merge_three_estimators():
    import pandas as pd
    import skrub

    from skrub import TableVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor

    # --------------------
    # Input data
    # --------------------
    df_main = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "country": ["US", "IT", "US", "FR"],
        "age": [25, 40, 35, 50],
    })

    df_aux = pd.DataFrame({
        "country": ["US", "IT", "FR"],
        "gdp": [70000, 35000, 42000],
    })

    df_y = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "salary": [60000, 50000, 65000, 55000],
    })

    # --------------------
    # skrub Vars
    # --------------------
    main = skrub.var("main", df_main)
    aux = skrub.var("aux", df_aux)
    y = skrub.var("y", df_y)["salary"]

    # --------------------
    # Pipeline
    # --------------------
    X = (
        main
        .merge(aux, on="country", how="left")
        .drop("id", axis=1)
    )

    X_vec = X.skb.apply(TableVectorizer())
    print("X_vec")
    print(X_vec)
    X_scaled = X_vec.skb.apply(StandardScaler())
    print("X_scaled")
    print(X_scaled)
    X_pca = X_scaled.skb.apply(PCA(n_components=2))
    print("X_pca")
    print(X_pca)

    predictor = X_pca.skb.apply(
        HistGradientBoostingRegressor(),
        y=y,
    )
    print(predictor)

    # --------------------
    # Execute
    # --------------------
    # Note!
    # the solution does not support .skb.eval()
    # result = predictor.skb.eval()
    # print(result)

def main_merge_isin_agg_complex():
    # not yet supported
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

    joined = (
        main_table
        .merge(aux_table, on="Country", how="inner")
    )

    # ---- isin filtering stage ----
    european_countries = ["Italy", "Belgium"]

    filtered = joined[
        joined["Country"].isin(european_countries)
    ]

    # ---- aggregation after isin ----
    # raise NotImplementedError("""The solution for agg does not support specification of parameters in this way 
    #    .agg(total_population=("population", "sum"),
    #     city_count=("City", "nunique"))""")
    result = filtered.agg(
        total_population=("population", "sum"),
        city_count=("City", "nunique"),
    )

    print(result)
    print("#"*80)
    print("now with the evaluated prov_column")
    print(evaluate_provenance(result.skb.preview()))

    print()
    print("now by passing a dataop")
    print()
    print(evaluate_provenance(result))


def main_merge_isin_agg_complex2():
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

    # ---- isin filtering stage ----
    european_countries = ["Italy", "Belgium"]
    filtered = joined[joined["Country"].isin(european_countries)]

    # ---- dictionary-based aggregation (equivalent to keyword style) ----
    raise NotImplementedError("""The solution for agg does not support specification of parameters in this way """)
    agg_dict = {
        "total_population": ("population", "sum"),
        "city_count": ("City", "nunique")
    }

    # Perform aggregation
    result = filtered.agg(agg_dict)

    print(result)


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
    print(joined)

    # ---- isin filtering stage ----
    european_countries = ["Italy", "Belgium"]
    filtered = joined[joined["Country"].isin(european_countries)]

    print("filtered")
    print(filtered)

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
    print(result)

def main_merge_isin_agg_simpler_projection():
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
    print(joined)

    # ---- isin filtering stage ----
    european_countries = ["Italy", "Belgium"]
    filtered = joined[joined["Country"].isin(european_countries)]

    print("filtered")
    print(filtered)

    # ---- aggregation without creating new column names ----
    # TODO: if someone grabs any column -> send prov columns with it!
    raise NotImplementedError(""" Getattr -> when gets one column -> gets also all _prov* """)
    result = filtered[["population", "City"]].agg({
        "population": "sum",   # sum the population column
        "City": "nunique"      # count unique cities
    })
    
    print(result)



from sklearn.base import BaseEstimator, TransformerMixin, is_outlier_detector, is_classifier, is_regressor

def main_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    print("is_classifier(LogisticRegression())")
    print(is_classifier(LogisticRegression())) # -> True
    print("is_regressor(LogisticRegression())")
    print(is_regressor(LogisticRegression())) 

def main():
    from monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
    enable_why_data_provenance()


    
    main_merge()
    # main_agg()
    # main_left_merge_agg()
    # main_inner_merge_agg()
    # main_left_merge_groupby_agg()
    # main_inner_merge_groupby_agg()
    # main_inner_merge_groupby_agg_eval()
    # main_merge_merge()
    # main_merge_merge_agg()
    # main_merge_merge_agg2()
    # main_merge_merge_agg3()
    # main_merge_merge_agg4()
    # main_inner_merge_groupby_agg_merge()
    # main_merge_tablevectorizer()

    # main_dataops_intro_preview()
    # main_dataops_intro_eval()
    # main_logistic_regression()
    # main_merge_three_estimators()
    
    # both not supported
    main_merge_isin_agg_complex()
    # main_merge_isin_agg_complex2()

    # main_merge_isin_agg_simpler()

# python -m monkey_patching_v02.data_provenance.testing_integer_ids_provenance

if __name__ == "__main__":
    main()