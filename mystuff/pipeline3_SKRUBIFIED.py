#Scenario:
#Your company sells products online, and each product belongs to a category 
# (e.g., "Electronics", "Home Appliances", "Books"). However, because categories are entered manually, 
# there are many typos and inconsistent naming:

#"Electronics" → "Elecronics", "Electonics", "Eletronics"

#"Home Appliances" → "Home Applianes", "Home Appliences"

#"Books" → "Boks", "Book"

#These inconsistencies make reporting, grouping, and ML analysis harder.

#We want to clean the category column automatically before analyzing sales, computing product popularity, or 
# feeding the data to a machine learning model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skrub
from skrub import deduplicate
from skrub import TableVectorizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
customers = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_customers_dataset.csv')
orders = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_orders_dataset.csv')
order_items =  pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_items_dataset.csv')
payments = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv')
reviews = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_reviews_dataset.csv')
order_payments = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv')
df = geolocation = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_geolocation_dataset.csv')
products = pd.read_csv('C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_products_dataset.csv')

# --- Detect categorical columns ---
# We'll consider object dtype columns as categorical
df = df.sample(frac=0.01)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Detected categorical columns:", categorical_cols)

# --- Exclude ID and timestamp columns ---
safe_categorical_cols = [
    col for col in categorical_cols
    if ("_id" not in col.lower()) and ("date" not in col.lower()) and ("timestamp" not in col.lower())
]
print("Columns safe to deduplicate:", safe_categorical_cols)
df_skrub = skrub.var("DF_skrub", df)
i = 0
while i < len(safe_categorical_cols):
    collumn = safe_categorical_cols[i]
    print(collumn)
    df_skrub.assign(collumn = df[safe_categorical_cols[i]])
    df_skrub[safe_categorical_cols[i]].skb.apply_func(deduplicate)
    print(df_skrub)
    i = i + 1
print(df_skrub)
# --- Deduplicate safe categorical columns ---

    
   


# --- Optional: Save cleaned DataFrame ---
# df.to_csv("cleaned_dataset.csv", index=False)