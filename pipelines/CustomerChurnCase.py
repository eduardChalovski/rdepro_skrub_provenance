import time
start_time = time.time()
import sys
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
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import skrub
from skrub import SquashingScaler, ToDatetime
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

customers = skrub.var("customers", pd.read_csv('./src/datasets/olist_customers_dataset.csv').sample(frac = 0.001))
orders = skrub.var("orders", pd.read_csv('./src/datasets/olist_orders_dataset.csv').sample(frac = 0.001))
order_items = skrub.var("order_items", pd.read_csv('./src/datasets/olist_order_items_dataset.csv').sample(frac = 0.001))
payments = skrub.var("payments", pd.read_csv('./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.001))
reviews = skrub.var("reviews", pd.read_csv('./src/datasets/olist_order_reviews_dataset.csv').sample(frac = 0.001))
order_payments = skrub.var("order_payments", pd.read_csv('./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.001))
geolocation = skrub.var("geolocation", pd.read_csv('./src/datasets/olist_geolocation_dataset.csv').sample(frac = 0.001))

print("Files read, starting the preprocessing")

orders_full = (
    orders
    .merge(order_items, on='order_id', how='left')
    .merge(order_payments, on='order_id', how='left')
    .merge(reviews.skb.select(['order_id', 'review_score']), on='order_id', how='left')
)

orders_full = orders_full.skb.apply(ToDatetime(), cols = ['order_delivered_customer_date', 'order_estimated_delivery_date'])
orders_full = orders_full.assign(
    delivery_delay_days=(
        (orders_full['order_delivered_customer_date'] - orders_full['order_estimated_delivery_date'])
        .dt.total_seconds() / 86400
    ).fillna(0),
)

customer_features = orders_full.groupby('customer_id').agg(
    n_orders=('order_id', 'nunique'),
    sum_payment=('payment_value', 'sum'),
    mean_freight=('freight_value', 'mean'),
    mean_delivery_delay=('delivery_delay_days', 'mean'),
    worst_review_score=('review_score', 'min'),
).reset_index()

for col in customer_features.skb.preview().columns.drop('customer_id'):
    customer_features = customer_features.assign(**{col: customer_features[col].fillna(0)})

geo_features = customers.skb.select(['customer_id', 'customer_zip_code_prefix']).merge(
    geolocation.drop_duplicates('geolocation_zip_code_prefix'),
    left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix',
    how='left'
)
customer_features = customer_features.merge(
    geo_features.skb.select(['customer_id', 'geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state']),
    on='customer_id',
    how='left'
)

customer_features = customer_features.assign(
    churn_risk=(customer_features['worst_review_score'] <= 2).astype(int)
)

print(f"Churn risk distribution:\n{customer_features['churn_risk'].value_counts()}")

numeric_features = [
    'n_orders', 'sum_payment', 'mean_freight', 'mean_delivery_delay',
    'geolocation_lat', 'geolocation_lng',
]
categorical_features = ['geolocation_city', 'geolocation_state']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('squash', SquashingScaler()),     # compress outliers
            ('scale', RobustScaler())          # robust to remaining outliers (vs StandardScaler in pipeline1)
        ]), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ]
)

feature_cols = numeric_features + categorical_features

y = customer_features['churn_risk'].skb.mark_as_y()
Xpre = customer_features.skb.select(feature_cols)
X = (Xpre.skb.apply(preprocessor)).skb.mark_as_X()

model = HistGradientBoostingClassifier(random_state=0)
predictor = X.skb.apply(model, y=y)

learner = predictor.skb.make_learner(fitted=True)
split = predictor.skb.train_test_split(random_state=0)
score = learner.score(split["test"])
print(f"Test accuracy: {score}")

pred = learner.predict(split["test"])
print(pred)
end_time = time.time() 

elapsed = end_time - start_time
print(f"⏱ Elapsed time: {elapsed:.2f} seconds")