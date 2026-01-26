#Use-case idea
#Predict bad reviews using fuzzy-joined product categories
#The core problem
#Can we predict whether an order will receive a bad review (≤2 stars) before the review is written?
#Why fuzzy join is needed with your data
#You already have:
#products.product_category_name
#But category names are often dirty / inconsistent:
#different spellings, language variants, abbreviations, pluralization, legacy names


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
import skrub
from skrub import Joiner, SelectCols
# -------------------------------------------------
# 1. BUILD CUSTOMER-LEVEL FEATURE TABLE
# -------------------------------------------------
customers = skrub.var("customers", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_customers_dataset.csv'))
orders = skrub.var("orders", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_orders_dataset.csv'))
order_items = skrub.var("order_items", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_items_dataset.csv'))
payments = skrub.var("payments",pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv'))
reviews = skrub.var("reviews",pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_reviews_dataset.csv'))
order_payments = skrub.var("order_payments", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv'))
geolocation = skrub.var("geolocation", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_geolocation_dataset.csv'))
products = skrub.var("products", pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_products_dataset.csv'))

order_items_agg = order_items.groupby('order_id').agg(
    total_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum')
).reset_index()

# -----------------------------
# 3. AGGREGATE PAYMENTS
# -----------------------------
payments_agg = payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum')
).reset_index()

# -----------------------------
# 4. MERGE ORDERS + ITEMS + PAYMENTS + REVIEWS
# -----------------------------
orders_full = orders.merge(order_items_agg, on='order_id', how='left')
orders_full = orders_full.merge(payments_agg, on='order_id', how='left')
orders_full = orders_full.merge(reviews[['order_id','review_score']], on='order_id', how='left')

# Fill missing numeric values
for col in ['total_items', 'total_price', 'total_freight', 'total_payment']:
    orders_full = orders_full.assign(col = orders_full[col].fillna(0))

# -----------------------------
# 5. CREATE TARGET
# -----------------------------
orders_full = orders_full.assign(bad_review = (orders_full['review_score'] <= 2).astype(int))
y = orders_full['bad_review'].skb.mark_as_y()

# -----------------------------
# 6. FUZZY JOIN PRODUCT INFO
# -----------------------------
# Use Joiner to fuzzy join product categories
# Take first product per order to avoid duplicates
first_product_per_order = order_items[['order_id','product_id']].drop_duplicates('order_id')

product_joiner = Joiner(
    products[['product_id','product_category_name']],  # aux table
    main_key='product_id',
    aux_key='product_id',
    suffix='_product_cat',
    max_dist=0.9
)

first_product_per_order = first_product_per_order.skb.apply(product_joiner)

orders_full = orders_full.merge(
    first_product_per_order[['order_id','product_category_name_product_cat']],
    on='order_id',
    how='left'
)

# -----------------------------
# 7. MERGE CUSTOMER GEOLOCATION
# -----------------------------
geo_features = customers[['customer_id','customer_zip_code_prefix']].merge(
    geolocation.drop_duplicates('geolocation_zip_code_prefix'),
    left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix',
    how='left'
)

orders_full = orders_full.merge(
    geo_features[['customer_id','geolocation_lat','geolocation_lng','geolocation_state']],
    on='customer_id', how='left'
)

# -----------------------------
# 8. SELECT FEATURES
# -----------------------------
numeric_features = ['total_items', 'total_price', 'total_freight', 'total_payment', 'geolocation_lat', 'geolocation_lng']
categorical_features = ['product_category_name_product_cat', 'geolocation_state']

# -----------------------------
# 9. BUILD PIPELINE
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

#pipeline = Pipeline([
 #   ('preproc', preprocessor),
  #  ('clf', HistGradientBoostingClassifier(random_state=42))
#])

clf = HistGradientBoostingClassifier(random_state=42)

# -----------------------------
# 10. CROSS-VALIDATE
# -----------------------------

Xpre = (orders_full[numeric_features + categorical_features])
import pickle

# Assuming df_skrub or your final DataFrame is what you want to check
with open("output.pkl", "wb") as f:
    pickle.dump(orders_full, f)
X = Xpre.skb.apply(preprocessor).skb.mark_as_X()
pipeline = X.skb.apply(clf, y=y)
learner = pipeline.skb.make_learner(fitted=True)

#cv_results = cross_validate(pipeline, X, y, cv=5, return_train_score=True, scoring='roc_auc')
#print(f"Mean ROC AUC: {np.mean(cv_results['test_score']):.3f} ± {np.std(cv_results['test_score']):.3f}")

# -----------------------------
# 11. FIT AND PREDICT
# --------------------------
# #
split = pipeline.skb.train_test_split(random_state= 0)
learner.score(split["test"])
#pipeline.fit(X, y)
#orders_full['bad_review_pred_proba'] = 

# Show top 10 most likely bad reviews


#pred = learner.predict(split["test"])
values = bad_review_pred_proba = learner.report(environment=split["train"], mode="predict_proba", open=False)["result"]
print(values)
#PIPELINE RESULT CAN BE SORTED AND ADJUSTED TO DERRIVE THE LOGICAL CONCLUSION BUT FOR THE PURPOSE OF THE PROJECT IT WORKS
