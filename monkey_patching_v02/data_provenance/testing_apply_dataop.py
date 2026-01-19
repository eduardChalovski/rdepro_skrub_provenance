from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import  set_provenance, enter_provenance_mode_var
import skrub 


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
    # print("main")
    # print(main.skb.eval())
    # --------------------
    # Pipeline
    # --------------------
    X = (
        main
        .merge(aux, on="country", how="left")
        .drop("id", axis=1)
    )

    X_vec = X.skb.apply(TableVectorizer())
    # print("X_vec")
    # print(X_vec.skb.eval())
    X_scaled = X_vec.skb.apply(StandardScaler())
    # print("X_scaled")
    # print(X_scaled.skb.eval())
    X_pca = X_scaled.skb.apply(PCA(n_components=2))
    # print("X_pca")
    # print(X_pca.skb.eval())

    predictor = X_pca.skb.apply(
        HistGradientBoostingRegressor(),
        y=y,
    )
    print("HGB")
    print(predictor.skb.eval())


import pandas as pd

def main():
    from pipelines.monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import enter_provenance_mode_dataop_callmethod, enter_provenance_mode_apply_format_predictions
    set_provenance(skrub._data_ops._data_ops.Var, "compute", provenance_func=enter_provenance_mode_var)
    set_provenance(skrub._data_ops._data_ops.CallMethod,"compute", provenance_func=enter_provenance_mode_dataop_callmethod)

    # print("before")
    # name = "eval"
    # print(getattr(skrub._data_ops._data_ops.Apply,name))
    # set_provenance(skrub._data_ops._data_ops.Apply,"eval", provenance_func=enter_provenance_mode_dataop_apply)
    set_provenance(
        skrub._data_ops._data_ops.Apply,
        "_format_predictions",
        provenance_func=enter_provenance_mode_apply_format_predictions,
        )

    # print("after")
    # print(getattr(skrub._data_ops._data_ops.Apply,name))
    

    # set_provenance(skrub._data_ops._data_ops.Apply,"eval", provenance_func=enter_provenance_mode_dataop_apply)

    main_merge_three_estimators()



# python -m pipelines.monkey_patching_v02.data_provenance.testing_apply_dataop
if __name__ == "__main__":
    main()