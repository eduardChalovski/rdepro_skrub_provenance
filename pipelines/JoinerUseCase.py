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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
import skrub
from skrub import Joiner
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
print("Libraries imported")

customers = skrub.var("customers", pd.read_csv(f'./src/datasets/olist_customers_dataset.csv').sample(frac = 0.01))
orders = skrub.var("orders", pd.read_csv(f'./src/datasets/olist_orders_dataset.csv').sample(frac = 0.01))
order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv').sample(frac = 0.01))
payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.01))
reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv').sample(frac = 0.01))
order_payments = skrub.var("order_payments", pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.01))
geolocation = skrub.var("geolocation", pd.read_csv(f'./src/datasets/olist_geolocation_dataset.csv').sample(frac = 0.01))
products = skrub.var("products", pd.read_csv(f'./src/datasets/olist_products_dataset.csv').sample(frac = 0.01))

order_items_agg = order_items.groupby('order_id').agg(
    total_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum')
).reset_index()

payments_agg = payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum')
).reset_index()

orders_full = orders.merge(order_items_agg, on='order_id', how='left')
orders_full = orders_full.merge(payments_agg, on='order_id', how='left')
orders_full = orders_full.merge(reviews.skb.select(['order_id','review_score']), on='order_id', how='left')

for col in ['total_items', 'total_price', 'total_freight', 'total_payment']:
    orders_full = orders_full.assign(col = orders_full[col].fillna(0))

orders_full = orders_full.assign(bad_review = (orders_full['review_score'] <= 2).astype(int))
y = orders_full['bad_review'].skb.mark_as_y()

first_product_per_order = order_items.skb.select(['order_id','product_id']).drop_duplicates('order_id')

product_joiner = Joiner(
    products.skb.select(['product_id','product_category_name']),  # aux table
    main_key='product_id',
    aux_key='product_id',
    suffix='_product_cat',
    max_dist=0.9
)

first_product_per_order = first_product_per_order.skb.apply(product_joiner)

orders_full = orders_full.merge(
    first_product_per_order.skb.select(['order_id','product_category_name_product_cat']),
    on='order_id',
    how='left'
)

geo_features = customers.skb.select(['customer_id','customer_zip_code_prefix']).merge(
    geolocation.drop_duplicates('geolocation_zip_code_prefix'),
    left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix',
    how='left'
)

orders_full = orders_full.merge(
    geo_features.skb.select(['customer_id','geolocation_lat','geolocation_lng','geolocation_state']),
    on='customer_id', how='left'
)


numeric_features = ['total_items', 'total_price', 'total_freight', 'total_payment', 'geolocation_lat', 'geolocation_lng']
categorical_features = ['product_category_name_product_cat', 'geolocation_state']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

clf = HistGradientBoostingClassifier(random_state=42)

Xpre = (orders_full.skb.select(numeric_features + categorical_features))
X = Xpre.skb.apply(preprocessor).skb.mark_as_X()
pipeline = X.skb.apply(clf, y=y)

print(pipeline.skb.draw_graph().open())
learner = pipeline.skb.make_learner(fitted=True)

split = pipeline.skb.train_test_split(random_state= 0)
learner.score(split["test"])

values = bad_review_pred_proba = learner.report(environment=split["train"], mode="predict_proba", open=False)["result"]
print(values)
#PIPELINE RESULT CAN BE SORTED AND ADJUSTED TO DERRIVE THE LOGICAL CONCLUSION BUT FOR THE PURPOSE OF THE PROJECT IT WORKS.
#WE DECIDED NOT TO INCLUDE THE FINAL CALCULATION AND MACHINE LEARNING TO SAVE TIME AND BECAUSE WE ANTED TO ONLY FOCUS ON
#PREPROCESSING
end_time = time.time() 
elapsed = end_time - start_time
print(f"⏱ Elapsed time: {elapsed:.2f} seconds")
