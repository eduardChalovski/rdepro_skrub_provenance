# Imbalanced Learning with RandomUnderSampler (Skrub + Provenance)
# Goal: predict late deliveries, and use a sampler that changes the number of rows.
#
# Run:
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
from sklearn.ensemble import HistGradientBoostingClassifier

# imbalanced-learn
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
    order_delivered_customer_date=pd.to_datetime(df["order_delivered_customer_date"], errors="coerce"),
    order_estimated_delivery_date=pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce"),
)

df = df.assign(
    is_late=(df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).fillna(False).astype(int)
)

# simple features
df = df.assign(
    payment_value=df["payment_value"].fillna(0),
    payment_installments=df["payment_installments"].fillna(0),
    price=df["price"].fillna(0),
    freight_value=df["freight_value"].fillna(0),
)

feature_cols = ["payment_value", "payment_installments", "price", "freight_value", "payment_type", "customer_state"]

Xraw = df.skb.select(feature_cols)
y = df["is_late"].skb.mark_as_y()

print("Target distribution:")
print(df["is_late"].value_counts(dropna=False))



numeric_features = ["payment_value", "payment_installments", "price", "freight_value"]
categorical_features = ["payment_type", "customer_state"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# We first transform X into numeric matrix, then apply undersampling, then fit the model.
# This keeps the code simple and makes it easier to see what happens with row counts.

X = Xraw.skb.apply(preprocessor).skb.mark_as_X()

dummy_model = HistGradientBoostingClassifier(random_state=0)
predictor = X.skb.apply(dummy_model, y=y)

split = predictor.skb.train_test_split(random_state=0)

X_train = split["train"]["_skrubimpl_X"]
y_train = split["train"]["_skrubimpl_y"]
X_test = split["test"]["_skrubimpl_X"]
y_test = split["test"]["_skrubimpl_y"]

print("Before undersampling:", X_train.shape, y_train.shape)

rus = RandomUnderSampler(random_state=0)
X_train2, y_train2 = rus.fit_resample(X_train, y_train)

print("After undersampling:", X_train2.shape, y_train2.shape)



model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_train2, y_train2)

score = model.score(X_test, y_test)
print(f"Test accuracy: {score}")
