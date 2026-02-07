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
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Detected categorical columns:", categorical_cols)

# --- Exclude ID and timestamp columns ---
safe_categorical_cols = [
    col for col in categorical_cols
    if ("_id" not in col.lower()) and ("date" not in col.lower()) and ("timestamp" not in col.lower())
]
print("Columns safe to deduplicate:", safe_categorical_cols)

# --- Deduplicate safe categorical columns ---
for col in ['geolocation_city', 'geolocation_state']:
    print(f"Deduplicating column: {col}")
    
    # Optionally, show number of unique values before dedupe
    unique_before = df[col].nunique()
    print(f"Unique values before dedupe: {unique_before}")
    print(len(df[col]))
    # Run deduplication
    cleaned = deduplicate(df[col])
    
    # Show number of unique values after dedupe
    unique_after = pd.Series(cleaned).nunique()
    print(f"Unique values after dedupe: {unique_after}\n")
    
    # Assign back to dataframe
    df[col + "_cleaned"] = cleaned

# --- Check results ---
for col in safe_categorical_cols:
    print(f"\nOriginal unique values in '{col}':", df[col].nunique())
    print(f"Deduplicated unique values in '{col}_cleaned':", df[col + "_cleaned"].nunique())

# --- Optional: Save cleaned DataFrame ---
# df.to_csv("cleaned_dataset.csv", index=False)