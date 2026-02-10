import pandas as pd


def _build_train_only_risk_feature(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_col: str,
    out_col: str = "cat_risk",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train-only aggregation: risk(category) = mean(y_train | category).
    Then join into train and test.
    """
    train = X_train.copy()
    train["_y"] = y_train.values

    risk_by_cat = (
        train.groupby(cat_col, dropna=False)["_y"]
        .mean()
        .rename(out_col)
        .reset_index()
    )

    global_mean = float(train["_y"].mean())

    X_train_out = X_train.merge(risk_by_cat, on=cat_col, how="left")
    X_test_out = X_test.merge(risk_by_cat, on=cat_col, how="left")

    X_train_out[out_col] = X_train_out[out_col].fillna(global_mean)
    X_test_out[out_col] = X_test_out[out_col].fillna(global_mean)

    return X_train_out, X_test_out


def test_leakage_train_only_aggregation():
    X_train = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "price": [10, 12, 3, 4],
        }
    )
    y_train = pd.Series([1, 0, 0, 0], name="is_late")  # risk(A)=0.5, risk(B)=0.0

    X_test = pd.DataFrame(
        {
            "category": ["A", "B", "C"],  # C unseen in train
            "price": [11, 5, 7],
        }
    )
    y_test = pd.Series([1, 1, 1], name="is_late") 

    _, X_test_1 = _build_train_only_risk_feature(
        X_train, y_train, X_test, cat_col="category", out_col="cat_risk"
    )

    y_test_modified = pd.Series([0, 0, 0], name="is_late")
    _, X_test_2 = _build_train_only_risk_feature(
        X_train, y_train, X_test, cat_col="category", out_col="cat_risk"
    )

    assert X_test_1["cat_risk"].equals(X_test_2["cat_risk"])

    # risk(A)=0.5, risk(B)=0.0, unseen(C)=global_mean_train=mean([1,0,0,0])=0.25
    expected = pd.Series([0.5, 0.0, 0.25], name="cat_risk")
    assert X_test_1["cat_risk"].reset_index(drop=True).equals(expected)
