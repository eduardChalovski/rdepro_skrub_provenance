
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
    print(filtered)
    result = filtered.groupby("Country").agg(
        total_population=("population", "sum"),
        # city_count=("City", "nunique"),

    )

    print(result)
    result = filtered.agg(
        total_population=("population", "sum"),
        # city_count=("City", "nunique"),

    )

    print(result)

def main():
    from monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
    enable_why_data_provenance()
    
    main_merge_isin_agg_complex()

if __name__ == "__main__":
    main()