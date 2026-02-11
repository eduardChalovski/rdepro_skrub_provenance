import time
start_time = time.time()
import sys
from pathlib import Path
import subprocess
def run_uv_sync():
    """Install dependencies via uv before running the rest of the pipeline"""
    try:
        subprocess.run([sys.executable, "-m", "uv", "sync"], check=True)
        print("✅ uv dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print("❌ uv install failed")
        print(e)
        sys.exit(1)
run_uv_sync()
print("Done!")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from skrub import fuzzy_join
import pandas as pd
import skrub
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

customers = skrub.var("customers", pd.read_csv(f'./src/datasets/olist_customers_dataset.csv').sample(frac = 0.01))
orders = skrub.var("orders", pd.read_csv(f'./src/datasets/olist_orders_dataset.csv').sample(frac = 0.01))
order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv').sample(frac = 0.01))
payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.01))
reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv').sample(frac = 0.01))
order_payments = skrub.var("order_payments", pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.01))
geolocation = skrub.var("geolocation", pd.read_csv(f'./src/datasets/olist_geolocation_dataset.csv').sample(frac = 0.01))

print(order_items.columns)
print(payments.columns)

augmented_df = fuzzy_join(
    order_items.skb.preview(), 
    payments.skb.preview(), 
    left_on="product_id",  
    right_on="order_id",  
    add_match_info=True,
)

print(augmented_df)
end_time = time.time() 

elapsed = end_time - start_time
print(f"⏱ Elapsed time: {elapsed:.2f} seconds")