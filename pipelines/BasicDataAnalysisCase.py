import sys
from pathlib import Path
import subprocess
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

# Run this first
run_uv_sync()
print("Done!")
print("Done!")
sys.path.append(str(Path(__file__).resolve().parents[1]))


# End-to-End Data Analysis
# =====================================================

"""
Objective:
Analyze Brazilian e-commerce orders to understand delivery performance,
customer satisfaction, and key operational drivers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import skrub
from skrub import ToDatetime
from skrub import TableReport
sns.set(style="whitegrid")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--track-provenance",
    action="store_true",
    help="Enable provenance tracking"
)
args = parser.parse_args()

if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance, evaluate_provenance
    enable_why_data_provenance()
else:
    print("Provenance is disabled")



customers = skrub.var("customers", pd.read_csv(f'./src/datasets/olist_customers_dataset.csv'))
orders =  pd.read_csv(f'./src/datasets/olist_orders_dataset.csv')
order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv'))
payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv'))
reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv'))

#df = (
 #   orders
  #  .merge(customers, on="customer_id", how="left")
   # .merge(order_items, on="order_id", how="left")
    #.merge(payments, on="order_id", how="left")
    #.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
#)
orders = skrub.var("df", orders)
df = orders.merge(customers, on="customer_id", how="left")
df = df.merge(order_items, on="order_id", how="left")
df = df.merge(payments, on="order_id", how="left")
df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

date_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

#for col in date_cols:
 #   df[col] = pd.to_datetime(df[col], errors="coerce")

toDateTimeEncoder = ToDatetime()
df= df.skb.apply(toDateTimeEncoder, cols= date_cols)

# Only delivered orders make sense for delay analysis
df = df[df["order_status"] == "delivered"]

#df["delivery_delay"] = (
 #   df["order_delivered_customer_date"] -
  #  df["order_estimated_delivery_date"]
#.dt.days

df = df.assign(delivery_delay = (
    df["order_delivered_customer_date"] -
    df["order_estimated_delivery_date"]
).dt.days)

# Note from Eddie: why don't we add df["freight_value"] if it was originally the case?
#df["order_value"] = df["price"] + df["freight_value"]
df = df.assign(order_value = df["price"])

#df["is_delayed"] = (df["delivery_delay"] > 0).astype(int)
df = df.assign(is_delayed = (df["delivery_delay"] > 0).astype(int))

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=[
    "order_value",
    "freight_value",
    "payment_installments",
    "review_score",
    "delivery_delay"
])

# Remove extreme outliers (realistic delivery window)
df = df[(df["delivery_delay"] >= -20) & (df["delivery_delay"] <= 60)]

df.skb.draw_graph()
#report = df.skb.full_report()
#report
#display(df["_skrub_provenance_"])
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model_df = df.skb.select([
    "freight_value",
    "price",
    "payment_installments",
    "order_value",
    "review_score",
    "is_delayed"
])

#X = model_df.drop("is_delayed", axis=1)
#y = model_df["is_delayed"]
# model_df = model_df.drop("_skrub_provenance_", axis=1)
X = model_df.drop("is_delayed", axis=1).skb.mark_as_X()
y = model_df["is_delayed"].skb.mark_as_y()

print(X.head())
