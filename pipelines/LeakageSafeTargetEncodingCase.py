# Leakage-Safe Target Encoding Pipeline (Skrub + Provenance)
# Goal: predict late delivery, with a train-only aggregated feature (no data leakage).
#
# Run:
#   python -m pipelines.LeakageSafeTargetEncodingCase --track-provenance

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
order_items = skrub.var("order_items", pd.read_csv("./src/datasets/olist_order_items_dataset.csv"))
products = skrub.var("products", pd.read_csv("./src/datasets/olist_products_dataset.csv"))
cat_tr = skrub.var("cat_tr", pd.read_csv("./src/datasets/product_category_name_translation.csv"))

print("Datasets loaded")


products_en = (
    products
    .merge(cat_tr, on="product_category_name", how="left")
    .rename(columns={"product_category_name_english": "product_category_en"})
)

df = (
    orders
    .merge(order_items, on="order_id", how="left")
    .merge(products_en.skb.select(["product_id", "product_category_en"]), on="product_id", how="left")
)

df = df.assign(
    order_delivered_customer_date=pd.to_datetime(df["order_delivered_customer_date"], errors="coerce"),
    order_estimated_delivery_date=pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce"),
)

df = df.assign(
    is_late=(df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).fillna(False).astype(int)
)

# keep a few features
df = df.assign(
    price=df["price"].fillna(0),
    freight_value=df["freight_value"].fillna(0),
)

# We'll predict is_late at order-item rows (simple); could be aggregated to order-level later.
y = df["is_late"].skb.mark_as_y()
Xraw = df.skb.select(["product_category_en", "price", "freight_value"]).skb.mark_as_X()


# We build a dummy predictor just to get a split object from skrub.
dummy_model = HistGradientBoostingClassifier(random_state=0)
predictor = Xraw.skb.apply(dummy_model, y=y)
split = predictor.skb.train_test_split(random_state=0)

X_train = split["train"]["_skrubimpl_X"]
y_train = split["train"]["_skrubimpl_y"]
X_test = split["test"]["_skrubimpl_X"]
y_test = split["test"]["_skrubimpl_y"]


train_with_y = X_train.assign(is_late=y_train)

cat_stats = (
    train_with_y
    .groupby("product_category_en")
    .agg(cat_late_rate=("is_late", "mean"))
    .reset_index()
)

global_rate = float(train_with_y["is_late"].mean())


X_train2 = X_train.merge(cat_stats, on="product_category_en", how="left")
X_test2 = X_test.merge(cat_stats, on="product_category_en", how="left")


X_train2["cat_late_rate"] = X_train2["cat_late_rate"].fillna(global_rate)
X_test2["cat_late_rate"] = X_test2["cat_late_rate"].fillna(global_rate)


feature_cols = ["price", "freight_value", "cat_late_rate"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("scale", StandardScaler()),
        ]), feature_cols),
    ],
    remainder="drop",
)

X_train_final = X_train2.skb.select(feature_cols).skb.apply(preprocessor)
X_test_final = X_test2.skb.select(feature_cols).skb.apply(preprocessor)

model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_train_final, y_train)

score = model.score(X_test_final, y_test)
print(f"Test accuracy: {score}")
