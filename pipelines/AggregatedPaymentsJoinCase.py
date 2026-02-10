import sys
import subprocess
print("Installing dependencies from uv.lock using PDM...")
def run_uv_sync():
    """Install dependencies via uv before running the rest of the pipeline"""
    try:
        # Use subprocess to run shell commands
        subprocess.run([sys.executable, "-m", "uv", "sync"], check=True)
        print("✅ uv dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print("❌ uv install failed")
        print(e)
        sys.exit(1)
run_uv_sync()
print("Done!")
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
payments = skrub.var("payments", pd.read_csv("./src/datasets/olist_order_payments_dataset.csv"))

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

# Fill missing values 
payments_agg = payments_agg.assign(
    total_payment=lambda d: d["total_payment"].fillna(0),
    n_payments=lambda d: d["n_payments"].fillna(0),
    max_installments=lambda d: d["max_installments"].fillna(0),
    payment_type_count=lambda d: d["payment_type_count"].fillna(0),
)

orders_full = orders.merge(payments_agg, on="order_id", how="left")

# Parse dates 
orders_full = orders_full.assign(
    order_delivered_customer_date=lambda d: pd.to_datetime(d["order_delivered_customer_date"], errors="coerce"),
    order_estimated_delivery_date=lambda d: pd.to_datetime(d["order_estimated_delivery_date"], errors="coerce"),
)

# Target: late delivery 
orders_full = orders_full.assign(
    is_late=lambda d: (d["order_delivered_customer_date"] > d["order_estimated_delivery_date"]).fillna(False).astype(int)
)

print("Join and target creation done")

numeric_features = ["total_payment", "n_payments", "max_installments", "payment_type_count"]
categorical_features = ["order_status"]

Xraw = orders_full.skb.select(numeric_features + categorical_features)
y = orders_full["is_late"].skb.mark_as_y()

# Make preprocessing robust and compatible with pandas output
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

model = HistGradientBoostingClassifier(random_state=0)
predictor = X.skb.apply(model, y=y)

split = predictor.skb.train_test_split(random_state=0)
learner = predictor.skb.make_learner(fitted=True)

score = learner.score(split["test"])
print(f"Test accuracy: {score}")
