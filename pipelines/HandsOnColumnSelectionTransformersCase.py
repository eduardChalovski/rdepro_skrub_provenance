# Hands-On with Column Selection and Transformers (Olist + Skrub DataOps + Provenance)
# Goal: predict whether an order will be late (delivered after estimated date).
#
# Inspired by skrub example "Hands-On with Column Selection and Transformers":
# apply different transformers depending on the selected columns (types / names).
#
# Run:
#   python -m pipelines.HandsOnColumnSelectionTransformersCase --track-provenance

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

import skrub
from skrub import DatetimeEncoder  

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--track-provenance", action="store_true", help="Enable provenance tracking")
args = parser.parse_args()

if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
        enable_why_data_provenance,
        evaluate_provenance,
    )
    enable_why_data_provenance()
else:
    print("Provenance is disabled")

print("Libraries imported")


customers = skrub.var("customers", pd.read_csv("./src/datasets/olist_customers_dataset.csv"))
orders = skrub.var("orders", pd.read_csv("./src/datasets/olist_orders_dataset.csv"))
order_items = skrub.var("order_items", pd.read_csv("./src/datasets/olist_order_items_dataset.csv"))
payments = skrub.var("payments", pd.read_csv("./src/datasets/olist_order_payments_dataset.csv"))
products = skrub.var("products", pd.read_csv("./src/datasets/olist_products_dataset.csv"))
cat_tr = skrub.var("cat_tr", pd.read_csv("./src/datasets/product_category_name_translation.csv"))

print("Files read, starting preprocessing")


products_en = (
    products
    .merge(cat_tr, on="product_category_name", how="left")
    .rename(columns={"product_category_name_english": "product_category_en"})
)

orders_full = (
    orders
    .merge(order_items, on="order_id", how="left")
    .merge(payments, on="order_id", how="left")
    .merge(products_en.skb.select(["product_id", "product_category_en", "product_weight_g", "product_photos_qty"]),
           on="product_id", how="left")
    .merge(customers.skb.select(["customer_id", "customer_state", "customer_city"]), on="customer_id", how="left")
)

# Parse datetimes
orders_full = orders_full.assign(
    order_purchase_timestamp=pd.to_datetime(orders_full["order_purchase_timestamp"], errors="coerce"),
    order_delivered_customer_date=pd.to_datetime(orders_full["order_delivered_customer_date"], errors="coerce"),
    order_estimated_delivery_date=pd.to_datetime(orders_full["order_estimated_delivery_date"], errors="coerce"),
)

# Target: late delivery
orders_full = orders_full.assign(
    is_late=(
        (orders_full["order_delivered_customer_date"] > orders_full["order_estimated_delivery_date"])
    ).fillna(False).astype(int)
)

# A couple of simple numeric features
orders_full = orders_full.assign(
    n_installments=orders_full["payment_installments"].fillna(0),
    payment_value=orders_full["payment_value"].fillna(0),
    price=orders_full["price"].fillna(0),
    freight_value=orders_full["freight_value"].fillna(0),
)

# IMPORTANT: avoid leakage by NOT using delivered/estimated dates in X
# We'll keep purchase timestamp (available at purchase time).
feature_cols = [
    "order_purchase_timestamp",
    "payment_type",
    "n_installments",
    "payment_value",
    "price",
    "freight_value",
    "product_category_en",
    "product_weight_g",
    "product_photos_qty",
    "customer_state",
    "customer_city",
]

# Use skrub select to ensure provenance columns are preserved when enabled
Xraw = orders_full.skb.select(feature_cols)
y = orders_full["is_late"].skb.mark_as_y()


datetime_cols = ["order_purchase_timestamp"]
numeric_cols = ["n_installments", "payment_value", "price", "freight_value", "product_weight_g", "product_photos_qty"]
categorical_cols = ["payment_type", "product_category_en", "customer_state", "customer_city"]

datetime_pipe = Pipeline([
    ("dt", DatetimeEncoder()),  # encodes a single datetime column into useful numeric features
])

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("dt", datetime_pipe, datetime_cols),
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
)

model = HistGradientBoostingClassifier(random_state=0)

X = (Xraw.skb.apply(preprocessor)).skb.mark_as_X()
predictor = X.skb.apply(model, y=y)

split = predictor.skb.train_test_split(random_state=0)

learner = predictor.skb.make_learner(fitted=True)
score = learner.score(split["test"])
print(f"Test accuracy: {score}")

# Optional: inspect provenance right before prediction (if enabled)
if args.track_provenance:
    try:
       
        print("Provenance sample (first rows):")
        prov = evaluate_provenance(split["test"]["_skrubimpl_X"])
        print(prov.head())
    except Exception as e:
        print("Could not evaluate provenance here:", repr(e))
