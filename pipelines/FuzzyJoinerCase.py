import pandas as pd
import numpy as np
import skrub
import sys
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--track-provenance",
    action="store_true",
    help="Enable provenance tracking"
)
from skrub import fuzzy_join
args = parser.parse_args()
if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance, evaluate_provenance
    enable_why_data_provenance()
else:
    print("Provenance is disabled")


customers = skrub.var("customers", pd.read_csv(f'./src/datasets/olist_customers_dataset.csv'))
orders = skrub.var("orders", pd.read_csv(f'./src/datasets/olist_orders_dataset.csv'))
order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv').sample(frac = 0.01))
payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac= 0.01))
reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv'))
geolocation = skrub.var("geolocation", pd.read_csv(f'./src/datasets/olist_geolocation_dataset.csv'))



print(order_items.columns)
print(payments.columns)

augmented_df = fuzzy_join(
    order_items.skb.preview(),  # our table to join
    payments.skb.preview(),  # the table to join with
    left_on="product_id",  # the first join key column
    right_on="order_id",  # the second join key column
    add_match_info=True,
)

print(augmented_df.columns)





