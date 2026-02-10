from __future__ import annotations

import time
import tracemalloc
import pandas as pd
import numpy as np

import skrub
from skrub import Joiner

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier




def _fillna_numeric(df: skrub.DataOp, cols: list[str]) -> skrub.DataOp:
    for c in cols:
        df = df.assign(**{c: df[c].fillna(0)})
    return df


def build_orders_features(
    customers: skrub.DataOp,
    orders: skrub.DataOp,
    order_items: skrub.DataOp,
    payments: skrub.DataOp,
    reviews: skrub.DataOp,
    geolocation: skrub.DataOp,
) -> tuple[skrub.DataOp, skrub.DataOp]:
    """
    Returns:
      orders_full: DataOp with engineered numeric features + customer_id/order_id
      y: marked-as-y target series
    """
    # aggregate items
    order_items_agg = order_items.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    ).reset_index()

    # aggregate payments
    payments_agg = payments.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
    ).reset_index()

    # join base tables
    orders_full = orders.merge(order_items_agg, on="order_id", how="left")
    orders_full = orders_full.merge(payments_agg, on="order_id", how="left")
    orders_full = orders_full.merge(
        reviews.skb.select(["order_id", "review_score"]),
        on="order_id",
        how="left",
    )

    orders_full = _fillna_numeric(
        orders_full,
        ["total_items", "total_price", "total_freight", "total_payment"],
    )

    # target
    orders_full = orders_full.assign(
        bad_review=(orders_full["review_score"] <= 2).astype(int)
    )
    y = orders_full["bad_review"].skb.mark_as_y()

    # customer geo join
    geo_features = customers.skb.select(["customer_id", "customer_zip_code_prefix"]).merge(
        geolocation.drop_duplicates("geolocation_zip_code_prefix"),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    orders_full = orders_full.merge(
        geo_features.skb.select(
            ["customer_id", "geolocation_lat", "geolocation_lng", "geolocation_state"]
        ),
        on="customer_id",
        how="left",
    )

    return orders_full, y


def build_category_reference(products: skrub.DataOp) -> skrub.DataOp:
    """
    Canonical category table.
    You can replace this with a real curated mapping table later.
    """
    cat_ref = (
        products.skb.select(["product_category_name"])
        .dropna()
        .drop_duplicates()
        .rename(columns={"product_category_name": "category_canonical"})
        .reset_index(drop=True)
    )
    # Optional: add an ID (useful later)
    cat_ref = cat_ref.assign(category_id=np.arange(1, 1 + len(cat_ref.skb.preview())))
    return cat_ref


def attach_category_baseline_exact(
    order_items: skrub.DataOp,
    products: skrub.DataOp,
) -> skrub.DataOp:
    """
    Baseline: exact join on product_id (NOT fuzzy on text).
    Take first product per order and bring the raw product_category_name.
    """
    first_product_per_order = (
        order_items.skb.select(["order_id", "product_id"])
        .drop_duplicates("order_id")
        .reset_index(drop=True)
    )

    # exact join product_id -> products.product_id
    prod_small = products.skb.select(["product_id", "product_category_name"])
    first_product_per_order = first_product_per_order.merge(
        prod_small, on="product_id", how="left"
    )
    # standardize output col name
    first_product_per_order = first_product_per_order.rename(
        columns={"product_category_name": "category_raw_exact"}
    )
    return first_product_per_order


def attach_category_fuzzy_on_text(
    order_items: skrub.DataOp,
    products: skrub.DataOp,
    category_ref: skrub.DataOp,
    max_dist: float = 0.2,
) -> skrub.DataOp:
    """
    True fuzzy join use-case:
      products.product_category_name (dirty-ish)  ~  category_ref.category_canonical (canonical)

    Output: per order_id, a canonical category column.

    Notes:
    - max_dist depends on the underlying distance used by skrub Joiner.
      Smaller => stricter match. You may tune.
    """
    first_product_per_order = (
        order_items.skb.select(["order_id", "product_id"])
        .drop_duplicates("order_id")
        .reset_index(drop=True)
    )

    # 1) exact join to get dirty category text per product_id
    prod_small = products.skb.select(["product_id", "product_category_name"])
    per_order = first_product_per_order.merge(prod_small, on="product_id", how="left")

    # 2) fuzzy join from dirty category text -> canonical reference
    # main_key: product_category_name (dirty)
    # aux_key : category_canonical (canonical)
    joiner = Joiner(
        category_ref.skb.select(["category_canonical", "category_id"]),
        main_key="product_category_name",
        aux_key="category_canonical",
        suffix="_canon",
        max_dist=max_dist,
    )
    per_order = per_order.skb.apply(joiner)

    # Keep only what we need
    out = per_order.skb.select(
        ["order_id", "category_canonical_canon", "category_id_canon"]
    ).rename(
        columns={
            "category_canonical_canon": "category_canonical_fuzzy",
            "category_id_canon": "category_id_fuzzy",
        }
    )
    return out



def make_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


def run_training_pipeline(
    orders_full: skrub.DataOp,
    y: skrub.DataOp,
    extra_cat_cols: list[str],
    random_state: int = 0,
) -> dict:
    """
    Returns a dict with:
      - learner_score
      - predict_proba_sample (first 10)
    """
    numeric_features = [
        "total_items", "total_price", "total_freight", "total_payment",
        "geolocation_lat", "geolocation_lng",
    ]
    categorical_features = ["geolocation_state"] + extra_cat_cols

    preproc = make_preprocessor(numeric_features, categorical_features)
    clf = HistGradientBoostingClassifier(random_state=42)

    Xpre = orders_full.skb.select(numeric_features + categorical_features)
    X = Xpre.skb.apply(preproc).skb.mark_as_X()

    pipeline = X.skb.apply(clf, y=y)

    split = pipeline.skb.train_test_split(random_state=random_state)
    learner = pipeline.skb.make_learner()   # keep it simple; no fitted=True
    learner.fit(split["train"])
    score = learner.score(split["test"])

    proba = learner.report(
        environment=split["test"],
        mode="predict_proba",
        open=False,
    )["result"]

    # proba might be np array / dataframe depending on skrub version
    try:
        sample = proba[:10]
    except Exception:
        sample = proba

    return {"learner_score": score, "predict_proba_sample": sample}


def pipeline_baseline_exact(
    customers, orders, order_items, payments, reviews, geolocation, products
) -> dict:
    orders_full, y = build_orders_features(customers, orders, order_items, payments, reviews, geolocation)

    per_order_cat = attach_category_baseline_exact(order_items, products)

    orders_full = orders_full.merge(
        per_order_cat.skb.select(["order_id", "category_raw_exact"]),
        on="order_id",
        how="left",
    )

    return run_training_pipeline(
        orders_full=orders_full,
        y=y,
        extra_cat_cols=["category_raw_exact"],
        random_state=0,
    )


def pipeline_fuzzy_category(
    customers, orders, order_items, payments, reviews, geolocation, products,
    max_dist: float = 0.2,
) -> dict:
    orders_full, y = build_orders_features(customers, orders, order_items, payments, reviews, geolocation)

    category_ref = build_category_reference(products)

    per_order_cat = attach_category_fuzzy_on_text(
        order_items=order_items,
        products=products,
        category_ref=category_ref,
        max_dist=max_dist,
    )

    orders_full = orders_full.merge(
        per_order_cat.skb.select(["order_id", "category_canonical_fuzzy"]),
        on="order_id",
        how="left",
    )

    return run_training_pipeline(
        orders_full=orders_full,
        y=y,
        extra_cat_cols=["category_canonical_fuzzy"],
        random_state=0,
    )


# -----------------------------
# Minimal benchmark harness (time + tracemalloc)
# Later we’ll move this into pytest-benchmark.
# -----------------------------
def bench_one(name, fn, repeats: int = 3):
    times = []
    mem_peaks = []

    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(dt)
        mem_peaks.append(peak / (1024 * 1024))  # MB

    return {
        "name": name,
        "time_mean_s": float(np.mean(times)),
        "time_std_s": float(np.std(times)),
        "peak_mem_mean_mb": float(np.mean(mem_peaks)),
        "peak_mem_std_mb": float(np.std(mem_peaks)),
        "last_output": out,
    }


def benchmark_baseline_vs_fuzzy(
    customers, orders, order_items, payments, reviews, geolocation, products
):
    b1 = bench_one(
        "baseline_exact_join",
        lambda: pipeline_baseline_exact(customers, orders, order_items, payments, reviews, geolocation, products),
        repeats=3,
    )
    b2 = bench_one(
        "fuzzy_join_category_text",
        lambda: pipeline_fuzzy_category(customers, orders, order_items, payments, reviews, geolocation, products, max_dist=0.2),
        repeats=3,
    )
    return pd.DataFrame([b1, b2])


# -----------------------------
# Example usage with your skrub.var(...) objects
# -----------------------------
result_df = benchmark_baseline_vs_fuzzy(customers, orders, order_items, payments, reviews, geolocation, products)
print(result_df[["name", "time_mean_s", "peak_mem_mean_mb"]])
print("Baseline score:", result_df.loc[result_df["name"]=="baseline_exact_join", "last_output"].iloc[0]["learner_score"])
print("Fuzzy score:", result_df.loc[result_df["name"]=="fuzzy_join_category_text", "last_output"].iloc[0]["learner_score"])
