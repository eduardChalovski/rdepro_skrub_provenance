import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
import skrub
from sklearn.pipeline import Pipeline
from skrub import SquashingScaler 


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


# -------------------------------------------------
# 1. BUILD CUSTOMER-LEVEL FEATURE TABLE
# -------------------------------------------------
print("Libraries imported")
# base_path = "C:/Users/eduar/Documents/RDEPro_testing/rdepro_skrub_provenance"
customers = skrub.var("customers", pd.read_csv(f'./src/datasets/olist_customers_dataset.csv'))
orders = skrub.var("orders", pd.read_csv(f'./src/datasets/olist_orders_dataset.csv'))
order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv'))
payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv'))
reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv'))
order_payments = skrub.var("order_payments", pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv'))
geolocation = skrub.var("geolocation", pd.read_csv(f'./src/datasets/olist_geolocation_dataset.csv'))

print("Files read, starting the preprocessing")
# --- 1. Aggregate order items ---
# order_items columns: ['order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value']
order_items_agg = order_items.groupby('order_id').agg(
    total_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
).reset_index()

sum_price =     order_items.groupby('order_id')["price"].sum()                  # is not supported  -> provenance information gets lost
sum_price_agg = order_items.groupby('order_id').agg({"price":sum})              # is supported      -> provenance information propagates correctly
are_expressions_equal = sum_price.equals(sum_price_agg["price"])
print("Expressions are equal:", are_expressions_equal)                          # returns True

print("sum_price")
print(sum_price)
print("sum_price_agg")
print(sum_price_agg)
print(sum_price.equals(sum_price_agg["price"]))