#Use case: Predicting product return risk in e-commerce
#Problem

#An e-commerce company wants to predict whether an order will be returned based on textual product information and
#customer-written fields.


#Here we use the multiple string encoders case
# pip install sentence_transformers

import sys
import subprocess
print("Installing dependencies from uv.lock using PDM...")
subprocess.check_call([sys.executable, "-m", "pdm", "install"])
print("Done!")
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from skrub import TableVectorizer, GapEncoder, StringEncoder, TextEncoder, MinHashEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
import skrub
import pandas as pd
import numpy as np

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


customers = skrub.var("customers", pd.read_csv(f'./src/datasets/olist_customers_dataset.csv').sample(frac = 0.001))
orders = skrub.var("orders", pd.read_csv(f'./src/datasets/olist_orders_dataset.csv').sample(frac = 0.001))
order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv').sample(frac = 0.001))
payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.001))
reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv').sample(frac = 0.001))
order_payments = skrub.var("order_payments", pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv').sample(frac = 0.001))
geolocation = skrub.var("geolocation", pd.read_csv(f'./src/datasets/olist_geolocation_dataset.csv').sample(frac = 0.001))
products = skrub.var("products", pd.read_csv(f'./src/datasets/olist_products_dataset.csv').sample(frac = 0.001))


orders_feat = orders

orders_feat = orders_feat.assign(is_late =  (
    orders_feat["order_delivered_customer_date"]
    > orders_feat["order_estimated_delivery_date"]
).astype(int)
)


orders_feat = orders_feat.skb.select([
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "is_late",
])


# orders_feat = orders_feat[[
#     "order_id",
#     "customer_id",
#     "order_purchase_timestamp",
#     "is_late",
# ]]

order_items_agg = (
    order_items
    .groupby("order_id")
    .agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )
    .reset_index()
)

payments_agg = (
    payments
    .groupby("order_id")
    .agg(
        payment_types=("payment_type", lambda x: " ".join(x)),
        max_installments=("payment_installments", "max"),
        total_payment_value=("payment_value", "sum"),
    )
    .reset_index()
)
order_products = (
    order_items
    .merge(products, on="product_id", how="left")
    .groupby("order_id")
    .agg(
        product_categories=("product_category_name", lambda x: " ".join(x.dropna())),
        mean_weight=("product_weight_g", "mean"),
        max_photos=("product_photos_qty", "max"),
    )
    .reset_index()
)
customer_features = customers.skb.select([
    "customer_id",
    "customer_city",
    "customer_state",
])

# customer_features = customers[[
#     "customer_id",
#     "customer_city",
#     "customer_state",
# ]]

df = (
    orders_feat
    .merge(order_items_agg, on="order_id", how="left")
    .merge(payments_agg, on="order_id", how="left")
    .merge(order_products, on="order_id", how="left")
    .merge(customer_features, on="customer_id", how="left")
)

X = df.drop(columns=["is_late"]).skb.mark_as_X()
y = df["is_late"].skb.mark_as_y()
vectorizerGap = TableVectorizer(high_cardinality=GapEncoder(n_components=30)) 
vectorizerHash = TableVectorizer(high_cardinality=MinHashEncoder(n_components=30))
text_encoder = TextEncoder(
    "sentence-transformers/paraphrase-albert-small-v2",
    device="cpu",
)

vectorizerText = TableVectorizer(high_cardinality= text_encoder)


gapPipe = X.skb.apply(vectorizerGap)
print("gapPipe provenance:")
print(gapPipe.columns)
print(gapPipe[:10])
gapPipe = gapPipe.skb.apply( HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        random_state=0,
    ), y = y)
hashPipe = X.skb.apply(vectorizerHash)
print("hashPipe provenacne:")
print(hashPipe.columns)
print(hashPipe[:10])
hashPipe = hashPipe.skb.apply( HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        random_state=0,
    ), y = y)
textPipe = X.skb.apply(vectorizerText)
print("textPipe provenance:")
print(textPipe.columns)
print(textPipe[:10])

textPipe = textPipe.skb.apply( HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        random_state=0,
    ), y = y)


#gapLearner = gapPipe.skb.make_learner(fitted=True)
#hashLearner = hashPipe.skb.make_learner(fitted=True)
#textLearner = textPipe.skb.make_learner(fitted=True)


gapResults = gapPipe.skb.cross_validate(cv=2)
textResults = hashPipe.skb.cross_validate(cv=2)
hashResults = textPipe.skb.cross_validate(cv=2)

results = []
results.append(("gap",gapResults))
results.append(("text",textResults))
results.append(("hash",hashResults))
from matplotlib import pyplot as plt
print("asdf")
def plot_box_results(named_results):
    fig, ax = plt.subplots()
    names, scores = zip(
        *[(name, result["test_score"]) for name, result in named_results]
    )
    ax.boxplot(scores)
    ax.set_xticks(range(1, len(names) + 1), labels=list(names), size=12)
    ax.set_ylabel("ROC AUC", size=14)
    plt.title(
        "AUC distribution across folds (higher is better)",
        size=14,
    )
    plt.show()
plot_box_results(results)

