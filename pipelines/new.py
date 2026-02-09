# Customer Churn Risk Classification Pipeline (Skrub + Provenance)
# "Which customers are at risk of churning, based on their purchasing patterns and satisfaction signals?"
#
# Uses the same Olist e-commerce tables as pipeline1 (revenue prediction),
# but targets a BINARY CLASSIFICATION task: did the customer leave a negative review (score <= 2)?
#
# To run:
#   python -m pipelines.pipeline2_churn_classification --track-provenance

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import skrub
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
# 1. LOAD RAW TABLES
# -------------------------------------------------
print("Libraries imported")

customers = skrub.var("customers", pd.read_csv('./src/datasets/olist_customers_dataset.csv'))
orders = skrub.var("orders", pd.read_csv('./src/datasets/olist_orders_dataset.csv'))
order_items = skrub.var("order_items", pd.read_csv('./src/datasets/olist_order_items_dataset.csv'))
payments = skrub.var("payments", pd.read_csv('./src/datasets/olist_order_payments_dataset.csv'))
reviews = skrub.var("reviews", pd.read_csv('./src/datasets/olist_order_reviews_dataset.csv'))
order_payments = skrub.var("order_payments", pd.read_csv('./src/datasets/olist_order_payments_dataset.csv'))
geolocation = skrub.var("geolocation", pd.read_csv('./src/datasets/olist_geolocation_dataset.csv'))

print("Files read, starting the preprocessing")


# -------------------------------------------------
# 2. MERGE RAW TABLES AT ORDER-ROW LEVEL (one join chain)
# -------------------------------------------------
orders_full = (
    orders
    .merge(order_items, on='order_id', how='left')
    .merge(order_payments, on='order_id', how='left')
    .merge(reviews.skb.select(['order_id', 'review_score']), on='order_id', how='left')
)

# Datetime features for delivery delay
#orders_full = orders_full.assign(
#    order_delivered_customer_date=pd.to_datetime(orders_full['order_delivered_customer_date']),
#    order_estimated_delivery_date=pd.to_datetime(orders_full['order_estimated_delivery_date']),
#    order_purchase_timestamp=pd.to_datetime(orders_full['order_purchase_timestamp']),
#)

orders_full = orders_full.skb.apply(toDatetimeEncoder, cols = ['order_delivered_customer_date', 'order_estimated_delivery_date', 'order_purchase_timestamp'])
orders_full = orders_full.assign(
    delivery_delay_days=(
        (orders_full['order_delivered_customer_date'] - orders_full['order_estimated_delivery_date'])
        .dt.total_seconds() / 86400
    ).fillna(0),
    delivery_time_days=(
        (orders_full['order_delivered_customer_date'] - orders_full['order_purchase_timestamp'])
        .dt.total_seconds() / 86400
    ).fillna(0),
)


# -------------------------------------------------
# 3. SINGLE CUSTOMER-LEVEL AGGREGATION
# -------------------------------------------------
customer_features = orders_full.groupby('customer_id').agg(
    n_orders=('order_id', 'nunique'),
    mean_payment=('payment_value', 'mean'),
    sum_payment=('payment_value', 'sum'),
    mean_items=('order_item_id', 'count'),       # total item rows per customer
    mean_order_price=('price', 'mean'),
    mean_freight=('freight_value', 'mean'),
    mean_installments=('payment_installments', 'mean'),
    n_distinct_sellers=('seller_id', 'nunique'),
    mean_delivery_delay=('delivery_delay_days', 'mean'),
    max_delivery_delay=('delivery_delay_days', 'max'),
    mean_delivery_time=('delivery_time_days', 'mean'),
    worst_review_score=('review_score', 'min'),
    mean_review_score=('review_score', 'mean'),
).reset_index()

# Fill NaN from customers with no items/payments/reviews
for col in customer_features.skb.preview().columns.drop('customer_id'):
    customer_features = customer_features.assign(**{col: customer_features[col].fillna(0)})


# -------------------------------------------------
# 8. MERGE CUSTOMER GEOLOCATION  (same as pipeline1)
# -------------------------------------------------
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


# -------------------------------------------------
# 9. DEFINE TARGET: churn risk = worst review score <= 2
# -------------------------------------------------
customer_features = customer_features.assign(
    churn_risk=(customer_features['worst_review_score'] <= 2).astype(int)
)

print(f"Churn risk distribution:\n{customer_features['churn_risk'].value_counts()}")


# -------------------------------------------------
# 10. PREPARE FEATURES FOR ML
# -------------------------------------------------
numeric_features = [
    'n_orders', 'mean_payment', 'sum_payment', 'mean_items',
    'mean_order_price', 'mean_freight', 'mean_installments', 'n_distinct_sellers',
    'mean_delivery_delay', 'max_delivery_delay', 'mean_delivery_time',
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


# -------------------------------------------------
# 11. BUILD SKRUB PIPELINE & EVALUATE
# -------------------------------------------------
# Drop the review scores from X so the model can't trivially leak the target
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