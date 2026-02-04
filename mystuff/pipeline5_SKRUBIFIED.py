
from skrub import TableVectorizer, GapEncoder, StringEncoder, TextEncoder, MinHashEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from skrub import Joiner
import skrub
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

customers = skrub.var("customers", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_customers_dataset.csv').sample(frac=0.01))
orders = skrub.var("orders", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_orders_dataset.csv').sample(frac=0.01))
order_items = skrub.var("order_items", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_items_dataset.csv').sample(frac=0.01))
payments = skrub.var("payments",pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv').sample(frac=0.01))
reviews = skrub.var("reviews",pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_reviews_dataset.csv').sample(frac=0.01))
order_payments = skrub.var("order_payments", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv').sample(frac=0.01))
geolocation = skrub.var("geolocation", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_geolocation_dataset.csv').sample(frac=0.01))
products = skrub.var("products", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_products_dataset.csv').sample(frac=0.01))

orders_feat = orders


geo_centroids = (
    geolocation
    .groupby(
        [
            "geolocation_zip_code_prefix",
            "geolocation_city",
            "geolocation_state",
        ],
        as_index=False
    )
    .agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
    })
)

geo_centroids = (
    geolocation
    .groupby(
        [
            "geolocation_zip_code_prefix",
            "geolocation_city",
            "geolocation_state",
        ],
        as_index=False
    )
    .agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
    })
)

join_customers = Joiner(
    customers,
    aux_key=["customer_id"],
    main_key=["customer_id"],
)
join_geolocation = Joiner(
    geo_centroids,
    aux_key=[
        "geolocation_zip_code_prefix",
        "geolocation_city",
        "geolocation_state",
    ],
    main_key=[
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
)

vectorizer = TableVectorizer()



orders = orders.copy()

orders = orders.assign(late = (
    orders["order_delivered_customer_date"]
    > orders["order_estimated_delivery_date"]
).astype(int))

X = orders.drop(columns=["late"]).skb.mark_as_X()
y = orders["late"].skb.mark_as_y()
pipeline = X.skb.apply(join_customers).skb.apply(join_geolocation).skb.apply(vectorizer)
pipeline = pipeline.skb.apply(HistGradientBoostingClassifier( max_depth=6,
        learning_rate=0.05,
        random_state=0,
    ), y = y)
scores = pipeline.skb.cross_validate(cv = 5)

print("CV scores:", scores)
print("Mean CV accuracy:", scores.mean())