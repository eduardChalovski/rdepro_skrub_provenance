#Where SquashingScaler fits (the key insight)
#REMOVE OUTLIERS
#“Which customers are likely to generate high future revenue, despite extreme or irregular purchasing behavior?”

# python -m cProfile -o pipeline1_SKRUBIFIED_with_provenance_pd_integer_array_profile.out .\mystuff\pipeline1_SKRUBIFIED.py --track-provenance
# python -m cProfile -o pipeline1_SKRUBIFIED_with_provenance_profile.out .\mystuff\pipeline1_SKRUBIFIED.py --track-provenance
# python -m cProfile -o pipeline1_SKRUBIFIED_with_provenance_frozenset_profile.out .\mystuff\pipeline1_SKRUBIFIED.py --track-provenance
# python -m cProfile -o pipeline1_SKRUBIFIED_without_provenance_profile.out .\mystuff\pipeline1_SKRUBIFIED.py

# "sort cumtime`nstats 200`nquit" | python -m pstats .\pipeline1_SKRUBIFIED_with_provenance_frozenset_profile.out
# "sort cumtime`nstats 200`nquit" | python -m pstats .\pipeline1_SKRUBIFIED_with_provenance_profile.out
# "sort cumtime`nstats 200`nquit" | python -m pstats .\pipeline1_SKRUBIFIED_without_provenance_profile.out

# snakeviz .\pipeline1_SKRUBIFIED_with_provenance_pd_integer_array_profile.out
# snakeviz .\pipeline1_SKRUBIFIED_with_provenance_frozenset_profile.out
# snakeviz .\pipeline1_SKRUBIFIED_with_provenance_profile.out
# snakeviz .\pipeline1_SKRUBIFIED_without_provenance_profile.out


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
import skrub
from sklearn.pipeline import Pipeline
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
# 1. BUILD CUSTOMER-LEVEL FEATURE TABLE
# -------------------------------------------------
print("Libraries imported")
base_path = "C:/Users/eduar/Documents/RDEPro_testing/rdepro_skrub_provenance"
customers = skrub.var("customers", pd.read_csv(f'{base_path}/src/datasets/olist_customers_dataset.csv'))
orders = skrub.var("orders", pd.read_csv(f'{base_path}/src/datasets/olist_orders_dataset.csv'))
order_items = skrub.var("order_items", pd.read_csv(f'{base_path}/src/datasets/olist_order_items_dataset.csv'))
payments = skrub.var("payments",pd.read_csv(f'{base_path}/src/datasets/olist_order_payments_dataset.csv'))
reviews = skrub.var("reviews",pd.read_csv(f'{base_path}/src/datasets/olist_order_reviews_dataset.csv'))
order_payments = skrub.var("order_payments", pd.read_csv(f'{base_path}/src/datasets/olist_order_payments_dataset.csv'))
geolocation = skrub.var("geolocation", pd.read_csv(f'{base_path}/src/datasets/olist_geolocation_dataset.csv'))

print("Files read, starting the preprocessing")
# --- 1. Aggregate order items ---
# order_items columns: ['order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value']
order_items_agg = order_items.groupby('order_id').agg(
    total_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
).reset_index()

# --- 2. Aggregate order payments ---
# order_payments columns: ['order_id', 'payment_sequential', 'payment_type', 'payment_installments', 'payment_value']
order_payments_agg = order_payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum')
).reset_index()

# --- 3. Merge aggregated info into orders ---
# orders columns: ['order_id', 'customer_id', ...timestamps...]
orders_full = orders.merge(order_items_agg, on='order_id', how='left')
orders_full = orders_full.merge(order_payments_agg, on='order_id', how='left')

# Fill NaN payments/items with 0

orders_full = orders_full.assign(total_items = orders_full['total_items'].fillna(0))
orders_full = orders_full.assign(total_price = orders_full['total_price'].fillna(0))
orders_full = orders_full.assign(total_freight = orders_full['total_freight'].fillna(0))
orders_full = orders_full.assign(total_payment = orders_full['total_payment'].fillna(0))
# --- 4. Aggregate features per customer ---
customer_features = orders_full.groupby('customer_id').agg(
    n_orders=('order_id', 'count'),
    mean_payment=('total_payment', 'mean'),
    sum_payment=('total_payment', 'sum'),
    mean_items=('total_items', 'mean'),
    sum_items=('total_items', 'sum'),
    mean_order_price=('total_price', 'mean'),
    mean_freight=('total_freight', 'mean')
).reset_index()

# --- 5. Merge customer geolocation ---
# geolocation columns: ['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state']
geo_features = customers.skb.select(['customer_id','customer_zip_code_prefix']).merge(
    geolocation.drop_duplicates('geolocation_zip_code_prefix'),
    left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix',
    how='left'
)
customer_features = customer_features.merge(
    geo_features.skb.select(['customer_id','geolocation_lat','geolocation_lng','geolocation_city','geolocation_state']),
    on='customer_id',
    how='left'
)

# --- 6. Prepare features for ML ---
numeric_features = ['n_orders','mean_payment','sum_payment','mean_items','sum_items',
                    'mean_order_price','mean_freight','geolocation_lat','geolocation_lng']
categorical_features = ['geolocation_city','geolocation_state']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('squash', SquashingScaler()),   # skrub scaler
            ('scale', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# --- 7. Target variable ---
# Since you don't have future revenue, we can use 'sum_payment' as a proxy
y = customer_features['sum_payment'].skb.mark_as_y()
Xpre = customer_features.skb.select(numeric_features + categorical_features)
X = (Xpre.skb.apply(preprocessor)).skb.mark_as_X()

# --- 8. Build pipeline ---
model = HistGradientBoostingRegressor()
predictor = X.skb.apply(model, y=y)

# predictor.skb.draw_graph().open()
# g = predictor.draw_graph()

# with open("graph.png", "wb") as f:
#     f.write(g.png)

learner = predictor.skb.make_learner(fitted=True)
split = predictor.skb.train_test_split(random_state= 0)
learner.score(split["test"])

pred = learner.predict(split["test"])
print(pred)   

# --- 9. Cross-validate ---
#cv_results = cross_validate(learner, X, y, cv=5, return_train_score=True)


# --- 10. Fit the pipeline on all data ---

# --- 11. Make predictions ---

#customer_features = customer_features.assign(predicted_sum_payment =  learner.predict(split["test"]))
#cv_results = customer_features['predicted_sum_payment'].skb.cross_validate(cv = 5, return_train_score = True)
#print(f"R2 score: mean={np.mean(cv_results['test_score']):.3f}, std={np.std(cv_results['test_score']):.3f}")
#print(f"mean fit time: {np.mean(cv_results['fit_time']):.3f} seconds")
#print(customer_features[['customer_id', 'predicted_sum_payment']].head(20))

# Uncomment to run the determenism check correctly
# import pickle
# with open("output.pkl", "wb") as f:
#     pickle.dump(customer_features, f)