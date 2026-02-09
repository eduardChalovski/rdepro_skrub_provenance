import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier, plot_importance
import shap
import warnings
import skrub
from skrub import ToDatetime
from skrub import DatetimeEncoder
warnings.filterwarnings("ignore")
#PROVENANCE MODULE
# from monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_dataop, enter_provenance_mode_var
# set_provenance(skrub._data_ops._evaluation,"evaluate", provenance_func=enter_provenance_mode_dataop)
# set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)
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

    
#SECOND PROV MODULE
#from skrub._data_ops._provenance import trace_row, explain_provenance
print("Libraries imported")
base_path = "C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance"

customers = pd.read_csv(f'./src/datasets/olist_customers_dataset.csv')
orders =  pd.read_csv(f'./src/datasets/olist_orders_dataset.csv')
order_items = order_items = skrub.var("order_items", pd.read_csv(f'./src/datasets/olist_order_items_dataset.csv'))
order_payments = skrub.var("payments",pd.read_csv(f'./src/datasets/olist_order_payments_dataset.csv'))
order_reviews = skrub.var("reviews",pd.read_csv(f'./src/datasets/olist_order_reviews_dataset.csv'))

# Preprocessing
#df = pd.merge(orders, customers, on='customer_id')
df = skrub.var("df", orders.merge(customers, on="customer_id"))
df = df[df['order_status'] == 'delivered']
#df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
toDatetimeEncoder = ToDatetime() 
df= df.skb.apply(toDatetimeEncoder, cols= 'order_purchase_timestamp')

latest_date = df['order_purchase_timestamp'].max()
cutoff = latest_date - timedelta(days=180)
print(df.agg(latest_latest_date = ("order_purchase_timestamp","max")))
last_order =  df.groupby('customer_unique_id').agg({'order_purchase_timestamp': 'max'}).reset_index()
last_order = last_order.assign(churn = (last_order['order_purchase_timestamp'] < cutoff).astype(int))

order_counts = df.groupby('customer_unique_id').size().reset_index(name='order_count')
price = order_items.groupby('order_id')['price'].sum().reset_index()
payment = order_payments.groupby('order_id')['payment_value'].sum().reset_index()

features = df.merge(last_order, on='customer_unique_id')
features = features.merge(order_counts, on='customer_unique_id')
features = features.merge(price, on='order_id', how='left') 
features = features.merge(payment, on='order_id', how='left')
features = features.skb.select(['customer_unique_id', 'order_count', 'price', 'payment_value', 'churn']).dropna()
features = features.groupby('customer_unique_id').agg({
    'order_count': 'max',
    'price': 'mean',
    'payment_value': 'mean',
    'churn': 'max'
}).reset_index()

Xpre =  features.drop(['customer_unique_id', 'churn'], axis=1)
y = features['churn'].skb.mark_as_y()

scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X = (Xpre.skb.apply(scaler)).skb.mark_as_X()
print(X.columns)
print(X.head())
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#y_proba = model.predict_proba(X_test)[:, 1] 
#CANNOT FIND SKRUB EQUIVALENT TO THIS
delayed = X.skb.apply(model, y=y)
split = delayed.skb.train_test_split(random_state= 0)
learner = delayed.skb.make_learner()
learner.fit(split["train"])
learner.score(split["test"])
pred = learner.predict(split["test"])

# Evaluation
#print(classification_report(y_test, y_pred))
print(classification_report(split["y_test"], pred))
#print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
roc_auc_score(split["y_test"], pred)
