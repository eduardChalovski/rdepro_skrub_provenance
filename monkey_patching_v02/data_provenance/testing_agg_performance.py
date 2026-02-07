import sys
from pathlib import Path

project_root = Path.cwd().parents[1]
sys.path.insert(0, str(project_root))

print("Added to sys.path:", project_root)


# Olist Churn Prediction with Complete Visual Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier, plot_importance      # i thought xgboost is defined already in scikit learn
import shap
import warnings
import skrub
from skrub import ToDatetime
from skrub import DatetimeEncoder
import cProfile, pstats
# warnings.filterwarnings("ignore")
#PROVENANCE MODULE

# enable_why_data_provenance()

# from monkey_patching_v02.data_provenance.monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_dataop, enter_provenance_mode_var
# set_provenance(skrub._data_ops._evaluation,"evaluate", provenance_func=enter_provenance_mode_dataop)
# set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)


# from functools import wraps
# def enter_skrub_learner_provenance(func):
#     @wraps(func)
#     def wrapper(*args,**kwargs):
        
#         # print("we are here")
#         # print("arguemnts")
#         for arg in args:
#             # print(arg)
#             # print()
#             if isinstance(arg, dict):
#                 # print("In the following dataframe, the _prov columns should be dropped")
#                 # print(arg['_skrub_X'])
#                 if "_skrub_X" in arg.keys():
#                     arg['_skrub_X'] = arg['_skrub_X'].drop(columns=[col for col in arg['_skrub_X'].columns if col.startswith("_prov")])
#         result = func(*args,**kwargs)
        
#         # TODO: maybe reattach _prov cols to arg later
#         # print("result is ", result)

#         return result # Just execute the function and get the result
#     return wrapper


# set_provenance(skrub._data_ops._estimator.SkrubLearner, "_eval_in_mode", enter_skrub_learner_provenance)


from skrub import SelectCols

# Load data
#order_items = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')
#order_payments = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')
#order_reviews = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
def run_pipeline():

    customers = pd.read_csv('C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_customers_dataset.csv')
    orders = pd.read_csv('C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_orders_dataset.csv')
    order_items = skrub.var("order_items", pd.read_csv('C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_items_dataset.csv'))
    order_payments = skrub.var("order_payments",pd.read_csv('C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_payments_dataset.csv'))
    order_reviews = skrub.var("order_reviews", pd.read_csv('C:/Users/eduar/Documents/RDEPro_github_clean/rdepro_skrub_provenance/monkey_patching_v02/data_provenance/kagglePipelines/data/datasets/olistbr/brazilian-ecommerce/versions/2/olist_order_reviews_dataset.csv'))

    # Preprocessing
    # df = pd.merge(orders, customers, on='customer_id')
    df = skrub.var("df", orders.merge(customers, on="customer_id"))#[:10_000])
    df = df[df['order_status'] == 'delivered']
    #df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    # toDatetimeEncoder = ToDatetime() 
    # df= df.skb.apply(toDatetimeEncoder, cols= 'order_purchase_timestamp')

    # latest_date = df['order_purchase_timestamp'].max()
    # cutoff = latest_date - timedelta(days=180)
    # last_order = df.groupby('customer_unique_id').agg({"order_purchase_timestamp":"max"}).reset_index()
    # #last_order['churn'] = (last_order['order_purchase_timestamp'] < cutoff).astype(int)
    # last_order = last_order.assign(churn = (last_order['order_purchase_timestamp'] < cutoff).astype(int))
    # display(last_order)

    # order_counts = df.groupby('customer_unique_id').size()#.reset_index(name='order_count')
    # order_counts


    new_df = df.assign(df_prov = np.arange(len(df.skb.eval())))
    # new_df = df.assign(df_prov = np.arange(len(df)))
    # new_df
    # around 41 seconds
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id').agg({"customer_unique_id":"size",
    #                                                                         "df_prov":list})
    # print(new_df_order_counts_agg_size)

    
    # around 77 seconds
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id').agg({"customer_unique_id":"size",
    #                                                                         "df_prov":"unique"})
    # print(new_df_order_counts_agg_size)

    # around 38 seconds
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id').agg({"customer_unique_id":"size",
    #                                                                         "df_prov":set})
    # print(new_df_order_counts_agg_size)

    # around 59 seconds
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id').agg({"customer_unique_id":"size",
    #                                                                         "df_prov": lambda x: np.array(x, dtype=np.int64).tolist()})
    # print(new_df_order_counts_agg_size)

    # 13.5 secs
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id').agg({"customer_unique_id":"size"})#.reset_index() #.agg({"customer_unique_id":"size",
    # #                                                                         "df_prov": np.array()})
    # print(new_df_order_counts_agg_size)

    # 44 seconds
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id')['customer_unique_id'].size()#.reset_index() #.agg({"customer_unique_id":"size",
    # #                                                                         "df_prov": np.array()})
    # new_df_order_counts_agg_prov = new_df.groupby('customer_unique_id')['df_prov'].apply(set)#.reset_index() #.agg({"customer_unique_id":"size",

    # print(pd.concat([new_df_order_counts_agg_size.skb.preview(), new_df_order_counts_agg_prov.skb.preview()], axis=1))

    # 46.159 seconds
    # new_df_order_counts_agg_size = new_df.groupby('customer_unique_id')['customer_unique_id'].size()#.reset_index() #.agg({"customer_unique_id":"size",
    # #                                                                         "df_prov": np.array()})
    # new_df_order_counts_agg_prov = new_df.groupby('customer_unique_id')['df_prov'].apply(list)#.reset_index() #.agg({"customer_unique_id":"size",

    # print(pd.concat([new_df_order_counts_agg_size.skb.preview(), new_df_order_counts_agg_prov.skb.preview()], axis=1))

    #         77099668 function calls (76160137 primitive calls) in 62.839 seconds
    new_df_order_counts_agg_size = new_df.groupby('customer_unique_id')['customer_unique_id'].size()#.reset_index() #.agg({"customer_unique_id":"size",
    #                                                                         "df_prov": np.array()})
    new_df_order_counts_agg_prov = new_df.groupby('customer_unique_id')['df_prov'].apply(np.array)#.reset_index() #.agg({"customer_unique_id":"size",

    print(pd.concat([new_df_order_counts_agg_size.skb.preview(), new_df_order_counts_agg_prov.skb.preview()], axis=1))

def main():
    with cProfile.Profile() as profile:
        for _ in range(10):
            run_pipeline()

    profiling_results = pstats.Stats(profile)
    profiling_results.sort_stats(pstats.SortKey.CUMULATIVE)

    profiling_results.print_stats(30)

if __name__ == "__main__":
    main()