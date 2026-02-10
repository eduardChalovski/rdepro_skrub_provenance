# Aggregated Payments Join Pipeline (Skrub + Provenance)
# Goal: predict whether an order will be delivered late using aggregated payment information.
#
# Run:
#   python -m pipelines.AggregatedPaymentsJoinCase --track-provenance

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


parser = argparse.ArgumentParser()
parser.add_argument("--track-provenance", action="store_true")
args = parser.parse_args()

if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
    enable_why_data_provenance()
else:
    print("Provenance is disabled")


orders = skrub.var(
    "orders",
    pd.read_csv("./src/datasets/olist_orders_dataset.csv")
)
payments = skrub.var(
    "payments",
    pd.read_csv("./src/datasets/olist_order_payments_dataset.csv")
)

print("Datasets loaded")



payments_agg = (
    payments
    .groupby("order_id")
    .agg(
        total_payment=("payment_value", "sum"),
        n_payments=("payment_value", "count"),
        max_installments=("payment_installments", "max"),
        payment_type_count=("payment_type", "nunique"),
    )
    .reset_index()
)

# Fill missing values (orders without payments)
for col in payments_agg.columns:
    if col != "order_id":
        payments_agg[col] = payments_agg[col].fillna(0)



orders_full = orders.merge(
    payments_agg,
    on="order_id",
    how="left"
)

# Parse dates
orders_full = orders_full.assign(
    order_delivered_customer_date=pd.to_datetime(
        orders_full["order_delivered_customer_date"], errors="coerce"
    ),
    order_estimated_delivery_date=pd.to_datetime(
        orders_full["order_estimated_delivery_date"], errors="coerce"
    ),
)

# Target: late delivery
orders_full = orders_full.assign(
    is_late=(
        orders_full["order_delivered_customer_date"]
        > orders_full["order_estimated_delivery_date"]
    ).fillna(False).astype(int)
)

print("Join and target creation done")


numeric_features = [
    "total_payment",
    "n_payments",
    "max_installments",
    "payment_type_count",
]

categorical_features = ["order_status"]

Xraw = orders_full.skb.select(numeric_features + categorical_features)
y = orders_full["is_late"].skb.mark_as_y()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X = Xraw.skb.apply(preprocessor).skb.mark_as_X()


model = HistGradientBoostingClassifier(random_state=0)
predictor = X.skb.apply(model, y=y)

split = predictor.skb.train_test_split(random_state=0)
learner = predictor.skb.make_learner(fitted=True)

score = learner.score(split["test"])
print(f"Test accuracy: {score}")
