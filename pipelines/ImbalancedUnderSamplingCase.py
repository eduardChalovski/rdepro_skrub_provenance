# Imbalanced Learning with RandomUnderSampler (Skrub + Provenance)
# Goal: predict late deliveries, and use a sampler that changes the number of rows.
#
# Run:
#   python -m pipelines.ImbalancedUnderSamplingCase
#   python -m pipelines.ImbalancedUnderSamplingCase --track-provenance

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import skrub

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.under_sampling import RandomUnderSampler


parser = argparse.ArgumentParser()
parser.add_argument("--track-provenance", action="store_true")
args = parser.parse_args()

if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
    enable_why_data_provenance()
else:
    print("Provenance is disabled")

orders = skrub.var("orders", pd.read_csv("./src/datasets/olist_orders_dataset.csv"))
order_items = skrub.var("order_items", pd.read_csv("./src/datasets/olist_order_items_dataset.csv"))
payments = skrub.var("payments", pd.read_csv("./src/datasets/olist_order_payments_dataset.csv"))
customers = skrub.var("customers", pd.read_csv("./src/datasets/olist_customers_dataset.csv"))

print("Datasets loaded")

df = (
    orders
    .merge(order_items, on="order_id", how="left")
    .merge(payments, on="order_id", how="left")
    .merge(customers.skb.select(["customer_id", "customer_state"]), on="customer_id", how="left")
)

df = df.assign(
    order_delivered_customer_date=lambda d: pd.to_datetime(d["order_delivered_customer_date"], errors="coerce"),
    order_estimated_delivery_date=lambda d: pd.to_datetime(d["order_estimated_delivery_date"], errors="coerce"),
)

df = df.assign(
    is_late=lambda d: (d["order_delivered_customer_date"] > d["order_estimated_delivery_date"]).fillna(False).astype(int)
)

# Simple features (also use lambdas)
df = df.assign(
    payment_value=lambda d: d["payment_value"].fillna(0),
    payment_installments=lambda d: d["payment_installments"].fillna(0),
    price=lambda d: d["price"].fillna(0),
    freight_value=lambda d: d["freight_value"].fillna(0),
    payment_type=lambda d: d["payment_type"].fillna("unknown"),
    customer_state=lambda d: d["customer_state"].fillna("unknown"),
)

feature_cols = [
    "payment_value",
    "payment_installments",
    "price",
    "freight_value",
    "payment_type",
    "customer_state",
]

Xraw = df.skb.select(feature_cols)
y = df["is_late"].skb.mark_as_y()

print("Target distribution (preview):")
try:
    # preview() forces evaluation
    print(df.skb.preview()["is_late"].value_counts(dropna=False))
except Exception as e:
    print("Could not compute preview distribution:", repr(e))

numeric_features = ["payment_value", "payment_installments", "price", "freight_value"]
categorical_features = ["payment_type", "customer_state"]


ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]), categorical_features),
    ],
    remainder="drop",
)

X = Xraw.skb.apply(preprocessor).skb.mark_as_X()


dummy_model = HistGradientBoostingClassifier(random_state=0)
predictor = X.skb.apply(dummy_model, y=y)
split = predictor.skb.train_test_split(random_state=0)

X_train = split["train"]["_skrub_X"]
y_train = split["train"]["_skrub_y"]
X_test = split["test"]["_skrub_X"]
y_test = split["test"]["_skrub_y"]

print("Before undersampling:", X_train.shape, y_train.shape)

rus = RandomUnderSampler(random_state=0)
X_train2, y_train2 = rus.fit_resample(X_train, y_train)

print("After undersampling:", X_train2.shape, y_train2.shape)

model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_train2, y_train2)

score = model.score(X_test, y_test)
print(f"Test accuracy: {score}")

