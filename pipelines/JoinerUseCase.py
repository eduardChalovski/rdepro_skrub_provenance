

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
import skrub
from skrub import Joiner, SelectCols


from monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance, evaluate_provenance
enable_why_data_provenance()

print("Libraries imported")
base_path = "C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance"

customers = skrub.var("customers", pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_customers_dataset.csv'))
orders = skrub.var("orders", pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_orders_dataset.csv'))
order_items = skrub.var("order_items", pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_items_dataset.csv'))
payments = skrub.var("payments",pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv'))
reviews = skrub.var("reviews",pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_reviews_dataset.csv'))
order_payments = skrub.var("order_payments", pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv'))
geolocation = skrub.var("geolocation", pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_geolocation_dataset.csv'))
products = skrub.var("products", pd.read_csv(f'{base_path}/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_products_dataset.csv'))

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
# print("y")
# print(y.skb.eval())


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

#pipeline = Pipeline([
 #   ('preproc', preprocessor),
  #  ('clf', HistGradientBoostingClassifier(random_state=42))
#])
# from sklearn.ensemble import HistGradientBoostingClassifier
# from functools import wraps

# orig_fit = HistGradientBoostingClassifier.fit

# def debug_fit(self, X, y=None, *args, **kwargs):
#     print("\n[DEBUG] HGB.fit called")

#     # --- X diagnostics ---
#     print("[DEBUG] X type:", type(X))
#     print("[DEBUG] X dtypes summary:")
#     print(X.dtypes.value_counts())

#     bad_dtypes = X.dtypes[X.dtypes == "object"]
#     if len(bad_dtypes):
#         print("[DEBUG] object dtype columns:", list(bad_dtypes.index)[:10])

#     # --- y diagnostics ---
#     print("[DEBUG] y type:", type(y))
#     if y is not None:
#         try:
#             import numpy as np
#             y_arr = np.asarray(y)
#             print("[DEBUG] y array shape:", y_arr.shape)
#             print("[DEBUG] y array dtype:", y_arr.dtype)
#             print("[DEBUG] y first element:", y_arr[0])
#         except Exception as e:
#             print("[DEBUG] y conversion failed:", e)

#     return orig_fit(self, X, y, *args, **kwargs)


# HistGradientBoostingClassifier.fit = debug_fit


clf = HistGradientBoostingClassifier(random_state=42)


Xpre = (orders_full.skb.select(numeric_features + categorical_features))
X = Xpre.skb.apply(preprocessor).skb.mark_as_X()
# print(evaluate_provenance(X))
# provX_cols = X.skb.preview().columns
# X = X.drop(columns=[col for col in provX_cols if col.startswith("_prov")])
pipeline = X.skb.apply(clf, y=y)

print(pipeline.skb.draw_graph().open())
# all_cols = pipeline.skb.preview()
# prov_cols = [col for col in all_cols if col.startswith("_prov")]
learner = pipeline.skb.make_learner(fitted=True)#fitted=True) # testing if it is because of fitted=True(fitted=True)
# print("the error is thrown because of the fitted= True")
#cv_results = cross_validate(pipeline, X, y, cv=5, return_train_score=True, scoring='roc_auc')
#print(f"Mean ROC AUC: {np.mean(cv_results['test_score']):.3f} ± {np.std(cv_results['test_score']):.3f}")

split = pipeline.skb.train_test_split(random_state= 0)
# learner.fit(split["train"])       #TODO: question: why we use fitted true and then split train test? don't we fit on both train and test splits?
learner.score(split["test"])
#pipeline.fit(X, y)
#orders_full['bad_review_pred_proba'] = 

# Show top 10 most likely bad reviews


#pred = learner.predict(split["test"])
# learner.predict
values = bad_review_pred_proba = learner.report(environment=split["train"], mode="predict_proba", open=False)["result"]
print(values)
#PIPELINE RESULT CAN BE SORTED AND ADJUSTED TO DERRIVE THE LOGICAL CONCLUSION BUT FOR THE PURPOSE OF THE PROJECT IT WORKS
