from __future__ import annotations
import pandas as pd
import skrub
from functools import wraps
# from provenance_utils_jeanne_performant import merge_with_provenance, groupby_aggregate_with_provenance, with_provenance
# from provenance_utils_jeanne_performant import PROV_COLUMN


# region Helpers
def get_dataop_dictionary(a_dataop):
    return a_dataop._skrub_impl.__dict__

def set_dataop_dictionary_val(a_dataop, attribute_name, new_val):
    a_dataop._skrub_impl.__dict__[attribute_name] = new_val

def get_var_dictionary(a_var):
    return a_var.__dict__

def set_var_dictionary_val(a_var, attribute_name, new_val):
    a_var.__dict__[attribute_name] = new_val


# region ProvenanceModule
class ProvenanceModule:
    def __init__(self):
        self.agg_func_over_prov_cols = list

    # region override agg
    def provenance_agg(self,a_dataop):
        # resulting_dataop_dictionary 
        d =  get_dataop_dictionary(a_dataop)

        preview_of_dataop_inside = get_dataop_dictionary(d["obj"])["results"]

        # ---- normalize agg args (FAST path, as before) ----
        args = d.get("args", ())
        kwargs = d.get("kwargs", {})

        if args:
            agg_dict = args[0]
        elif kwargs:
            agg_dict = kwargs
        else:
            agg_dict = {}
            d["args"] = (agg_dict,)
            d["kwargs"] = {}

        object_inside_preview = preview_of_dataop_inside["preview"]
        if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
            cols = object_inside_preview.obj.columns
            groupby_keys = object_inside_preview._grouper.names
        elif isinstance(object_inside_preview, pd.DataFrame):
            cols = object_inside_preview.columns
        else:
            cols = []

        if isinstance(agg_dict, str) or callable(agg_dict):
            # apply to all columns
            agg_dict = {col: agg_dict if not col.startswith("_prov") else self.agg_func_over_prov_cols for col in cols } #1
            if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
                for k in groupby_keys:
                    agg_dict.pop(k, None)
            
            d["args"] = (agg_dict,)
            d["kwargs"] = {}
            return a_dataop

        elif isinstance(agg_dict, list):
            agg_dict = {col: agg_dict if not col.startswith("_prov") else self.agg_func_over_prov_cols for col in cols} #2
            if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
                for k in groupby_keys:
                    agg_dict.pop(k, None)
            
            d["args"] = (agg_dict,)
            d["kwargs"] = {}
            return a_dataop
        
        elif isinstance(agg_dict, dict):
            # print("we enter this branch in the case of {'col_name':'max'} too")
            first_value = next(iter(agg_dict.values()), None)

            if isinstance(first_value, tuple):
                # print("agg_dict values are tuples (named aggregation style)")
                # example: {'new_col': ('old_col', 'max')}
                # or also (new_col_name = ('old_col', 'max'))
                # _prov0 = ("_prov0", list)
                # if cols:
                for col in cols:
                    if col.startswith("_prov"):
                        agg_dict[col] = (col, self.agg_func_over_prov_cols) # TODO: consider instead maybe slower but solves many issues prov_reduce_fast
                d["kwargs"] = agg_dict
                return a_dataop       
            else:
                # example: {'col_name':'max'}
                for col in cols:
                    if col.startswith("_prov"):
                        agg_dict[col] = self.agg_func_over_prov_cols
                
        d["args"] = (agg_dict,)
        d["kwargs"] = {}
        return a_dataop
    
PROVENANCE_MODULE = ProvenanceModule()  

from src.rdepro_skrub_provenance.provenance_utils import with_provenance_integers_shifted
from sklearn.base import is_outlier_detector, is_classifier, is_regressor

from skrub import selectors as s
PROV_SELECTOR = s.filter_names(str.startswith, "_prov")

# region prov entry point
def enter_provenance_mode_dataop(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        
        result_dataop = None
        
        for argument in args:
            if isinstance(argument, skrub.DataOp):
                result_dataop = argument
                break

        if isinstance(result_dataop, skrub.DataOp):                 # If that is a DataOp we can inspect what is stored inside
            final_dict = get_dataop_dictionary(result_dataop)                         # Inspecting
            if "method_name" in final_dict.keys():                                    # Having specific attribute in the dictionary classifies what kind of DataOp it is 
                # print(">>> THIS IS A CallMethod DataOp")
                # # ASPJ logic is covered here
                # # Pandas logic is covered here
                corresponding_provenance_function = getattr(PROVENANCE_MODULE, "provenance_"+final_dict["method_name"], None)
                if corresponding_provenance_function is not None:
                    corresponding_provenance_function(result_dataop)

            elif "estimator" in final_dict:
                # print(">>> THIS IS An Apply DataOp")

                est = final_dict["estimator"]
                if is_regressor(est) or is_classifier(est) or is_outlier_detector(est):
                    dataop_X_dict = get_dataop_dictionary(final_dict["X"])

                    preview  = dataop_X_dict["results"]["preview"]
                    prov_cols = [col for col in preview.columns if col.startswith("_prov")]
                    final_dict["X"] = final_dict["X"].drop(columns= prov_cols)
                    result = func(*args, **kwargs)

                    return result
                elif isinstance(est, skrub._select_cols.SelectCols):                    
                    final_dict["estimator"].__dict__["cols"] = final_dict["estimator"].__dict__["cols"] | PROV_SELECTOR

                else:
                    set_dataop_dictionary_val(a_dataop=result_dataop,
                                              attribute_name="cols",
                                              new_val=final_dict["cols"] - PROV_SELECTOR)
                

                
                # # print(" type of estimator =", type(final_dict["estimator"]))
            elif "attr_name" in final_dict:
                # TODO: introduce provenance, if one column is selected -> attach to it all prov_cols -> risky if for example ApplyToCols takes one column and gets a dataframe..
                pass
                # print(">>> THIS IS A GetAttr DataOp")
                # print("    attr_name =", final_dict["attr_name"])
                # print("    source_object =", final_dict.get("source_object"))
            else:
                pass
                # print("The evaluated DataOp is neither a CallMethod, nor a GetAttr, not a Apply")
                # print("# printing its dictionary")
                # print(final_dict)
            
            # print("[PROVENANCE]: END")
            ## print(final_dict)

        return func(*args, **kwargs) # Just execute the function and get the result
    return wrapper

# # region prov entry point
# def enter_provenance_mode_dataop_x(func):
#     @wraps(func)
#     def wrapper(*args,**kwargs):
        
#         result_dataop = None
        
#         print("[PROVENANCE]: Start")
        
#         for argument in args:
                     
            
#             if isinstance(argument, skrub.DataOp):
#                 result_dataop = argument
#                 break

#         if isinstance(result_dataop, skrub.DataOp):                 # If that is a DataOp we can inspect what is stored inside
#             final_dict = get_dataop_dictionary(result_dataop)                         # Inspecting
#             if "method_name" in final_dict.keys():                                    # Having specific attribute in the dictionary classifies what kind of DataOp it is 
#                 # print(">>> THIS IS A CallMethod DataOp")
#                 # # ASPJ logic is covered here
#                 # # Pandas logic is covered here
#                 # print("    method_name =", final_dict["method_name"])
#                 # print("    obj =", final_dict.get("obj"))
#                 # print("    args =", final_dict.get("args"))
#                 # print("    kwargs =", final_dict.get("kwargs"))
#                 corresponding_provenance_function = getattr(PROVENANCE_MODULE, "provenance_"+final_dict["method_name"], None)
#                 if corresponding_provenance_function is None:
#                     # print(f"""[PROVENANCE] Can't find a provenance_{final_dict["method_name"]} in the ProvenanceModule.""")
#                     pass
#                 else:
#                     result_df = corresponding_provenance_function(result_dataop)

#             elif "estimator" in final_dict:
#                 # print(">>> THIS IS An Apply DataOp")
#                 # print(argument)
#                 # print(final_dict)
#                 # print(final_dict.keys())
#                 # print(final_dict)
#                 # print("    name =", final_dict["name"])                
#                 print("    X =", final_dict["X"])
#                 print("    get_dataop_dictionary(X) =", get_dataop_dictionary(final_dict["X"]))
#                 # print("    y =", final_dict["y"])
#                 # print("    estimator =", final_dict["estimator"])

#                 est = final_dict["estimator"]
#                 if is_regressor(est) or is_classifier(est) or is_outlier_detector(est):

#                     dataop_X_dict = get_dataop_dictionary(final_dict["X"])
#                     preview  = dataop_X_dict["results"]["preview"]

#                     X_prov = s.select(preview, PROV_SELECTOR)
#                     X_main = s.select(preview, final_dict["cols"] - PROV_SELECTOR)

#                     # mutate preview before estimator runs
#                     dataop_X_dict["results"]["preview"] = X_main

#                     result = func(*args, **kwargs)

#                     X_out = pd.concat([result, X_prov], axis=1)

#                     # update Apply DataOp result (this is the critical part for propagation)
#                     set_dataop_dictionary_val(
#                         a_dataop=result_dataop,
#                         attribute_name="results",
#                         new_val={
#                             **final_dict["results"],
#                             "value": X_out,
#                             "preview": X_out,
#                         }
#                     )

#                     return X_out
#                 else:
#                     set_dataop_dictionary_val(a_dataop=result_dataop,
#                                               attribute_name="cols",
#                                               new_val=final_dict["cols"] - PROV_SELECTOR)
                

                
#                 # # print(" type of estimator =", type(final_dict["estimator"]))
#             elif "attr_name" in final_dict:
#                 # TODO: introduce provenance, if one column is selected -> attach to it all prov_cols -> risky if for example ApplyToCols takes one column and gets a dataframe..
#                 pass
#                 # print(">>> THIS IS A GetAttr DataOp")
#                 # print("    attr_name =", final_dict["attr_name"])
#                 # print("    source_object =", final_dict.get("source_object"))
#             else:
#                 pass
#                 # print("The evaluated DataOp is neither a CallMethod, nor a GetAttr, not a Apply")
#                 # print("# printing its dictionary")
#                 # print(final_dict)
            
#             print("[PROVENANCE]: END")
#             ## print(final_dict)

#         return func(*args, **kwargs) # Just execute the function and get the result
#     return wrapper

# region provenance var

def enter_provenance_mode_var(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        
        # result_dataop = None
        
        # print("[PROVENANCE Var]: Start")
        
        # print("printing result")
        # print(result)
        # print(type(result))
        # print("printing all arguments")
        for argument in args:
            # print("#"*40)
            # print("argument")
            # print(argument)
            # # print("isinstance of dataop")
            # # print(isinstance(argument, skrub.DataOp))
            # # print(isinstance(argument, skrub._data_ops._data_ops.DataOp))
            # # print(type(result_dataop) == skrub._data_ops._data_ops.DataOp)
            # # print(isinstance(argument, skrub._data_ops._data_ops.Var))
            # # print(type(argument)== skrub._data_ops._data_ops.Var)
            # print("type of argument")
            # print(type(argument))
            # print("#"*40)
            if isinstance(argument, skrub._data_ops._data_ops.Var):
                # print("I guess the CallMethod has var")
                # print("check worked!")
                # skrub.Var is the _skrub_impl -> to get the dictionary just call argument.__dict__
                # Note it uses get_var_dictionary, not get_dataop_dictionary 
                final_dict = get_var_dictionary(argument)
                # result_var = argument
                # if "name" in final_dict.keys() and "value" in final_dict.keys():
                #     # print(">>> THIS IS Var DataOp")
                #     # print(final_dict)
                #     set_var_dictionary_val(a_var=argument, 
                #                             attribute_name="value", # dataframe is stored under value -> we append the provenance column to it
                #                             new_val=with_provenance_integers_shifted(df=final_dict["value"], table_name=final_dict["name"]))
                    # print("Adjusted final_dict")
                    # print()
                    # print(final_dict)
            # if isinstance(argument, types.SimpleNamespace):
            #     print("#"*80)
            #     print("Found SimpleNamespace")
            #     print("#"*80)
            #     print(argument)
            #     print(argument.name)
            #     print("before:", argument)
            #     argument.__dict__["value"] = with_provenance_integers_shifted(df=final_dict["value"], table_name=final_dict["name"])
            #     print("after: ",argument)
                
        
        result = func(*args,**kwargs)
        # print(type(result))

        # In your setup, Sentinels mean:

        # “This value is intentionally not materialized yet.
        # It exists only so Skrub can track the graph.”
        # They appear during:
        # train_test_split
        # make_learner
        # fit_transform
        # any estimator boundary
        # They are expected and normal in Skrub.

        if isinstance(result, pd.DataFrame):
            result = with_provenance_integers_shifted(
                df=result,
                table_name=final_dict["name"]
            )
        # result = with_provenance_integers_shifted(df=final_dict["value"], table_name=final_dict["name"])


        # set_var_dictionary_val(
        #                     a_var=result_var,
        #                     attribute_name="results",
        #                     new_val={
        #                         **final_dict["results"],
        #                         "value": result,
        #                         "preview": result,
        #                     }
        #                 )
        # print(result)
        # print()
        # print("Result after adjusting SimpleNamespace")
        # print(result)
        
        # print("[PROVENANCE VAR]: End")

        return result # Just execute the function and get the result
    return wrapper


# def enter_provenance_mode_var(func):
#     @wraps(func)
#     def wrapper(*args,**kwargs):
        
#         for argument in args:
            
#             if isinstance(argument, skrub._data_ops._data_ops.Var):
                
#                 final_dict = get_var_dictionary(argument)          
        
#         result = func(*args,**kwargs)
#         result = with_provenance_integers_shifted(df=final_dict["value"], table_name=final_dict["name"])



#         return result # Just execute the function and get the result
#     return wrapper

def set_provenance(namespace, name_of_the_function, provenance_func=enter_provenance_mode_dataop):
    skrub_eval_namespace = namespace
    name = name_of_the_function
    skrub_eval = getattr(skrub_eval_namespace,name,None)
    setattr(skrub_eval_namespace, name, provenance_func(skrub_eval))
    # print(f"Set provenance for {name}")



# def prov_reduce_fast(values):
#         out = set()
#         for v in values:
#             if isinstance(v, (list, tuple, set)):
#                 out.update(v)
#             else:
#                 out.add(v)
#         return out

def set_reduce(values):
    # If pandas gives us a 1-column DataFrame, unwrap it
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    out = set()
    for v in values:
        if isinstance(v, (set, list, tuple)):
            out.update(v)
        else:
            out.add(v)
    return out


def list_reduce(values):
    # If pandas gives us a 1-column DataFrame, unwrap it
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    out = []
    append = out.append
    extend = out.extend

    for v in values:
        if isinstance(v, (list, tuple, set)):
            extend(v)
        else:
            append(v)

    return out

# region enable_provenance
def enable_why_data_provenance(agg_func_over_prov_cols=list):
    set_provenance(skrub._data_ops._evaluation,"evaluate", provenance_func=enter_provenance_mode_dataop)
    set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)
    
    if agg_func_over_prov_cols == "list_reduce":
        PROVENANCE_MODULE.agg_func_over_prov_cols = list_reduce
    elif agg_func_over_prov_cols == "set_reduce":
        PROVENANCE_MODULE.agg_func_over_prov_cols = set_reduce
    else:
        PROVENANCE_MODULE.agg_func_over_prov_cols = agg_func_over_prov_cols


    

    # from functools import wraps
    # def enter_skrub_learner_provenance(func):
    #     @wraps(func)
    #     def wrapper(*args,**kwargs):
            
    #         print("we are here")
    #         print("arguemnts")
    #         print(args)
    #         for arg in args[1:]:
    #             print(arg)
    #             print()
    #             if isinstance(arg, dict):
    #         #         pass
    #                 print("In the following dataframe, the _prov columns should be dropped")
    #                 # print(arg['_skrub_X'])
    #                 # if "_skrub_X" in arg.keys():
    #                 #     arg['_skrub_X'] = arg['_skrub_X'].drop(columns=[col for col in arg['_skrub_X'].columns if col.startswith("_prov")])
    #         # result = func(*args,**kwargs)
            
    #         # TODO: maybe reattach _prov cols to arg later
    #         # print("result is ", result)

    #         return func(*args,**kwargs) # Just execute the function and get the result
    #     return wrapper

    # set_provenance(skrub._data_ops._estimator.SkrubLearner, "_eval_in_mode", enter_skrub_learner_provenance)


import math

def normalize_cell(x):
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    return [x]

def show_provenance(df_with_many_prov_cols):
    """
    Receives a pandas dataframe with multiple _prov* cols 
    -> returns pd.DataFrame with one _prov column which is a set of all _prov* values
    
    :param df_with_many_prov_cols: pd.DataFrame with _prov column(s) or a skrub.DataOp
    """
    if isinstance(df_with_many_prov_cols, skrub.DataOp):
        df_with_many_prov_cols = df_with_many_prov_cols.skb.preview()

    prov_cols = [col for col in df_with_many_prov_cols.columns if col.startswith("_prov")] 
    return df_with_many_prov_cols[prov_cols]


# region evaluate_provenance
def evaluate_provenance(df_with_many_prov_cols):
    """
    Receives a pandas dataframe with multiple _prov* cols 
    -> returns pd.DataFrame with one _prov column which is a set of all _prov* values
    
    :param df_with_many_prov_cols: pd.DataFrame with _prov column(s) or a skrub.DataOp
    """
    if isinstance(df_with_many_prov_cols, skrub.DataOp):
        df_with_many_prov_cols = df_with_many_prov_cols.skb.preview()

    prov_cols = [col for col in df_with_many_prov_cols.columns if col.startswith("_prov")] 
    
    prov = df_with_many_prov_cols[prov_cols].map(normalize_cell)
    for col in prov_cols:
        while prov[col].map(lambda x: isinstance(x, list)).any():
            prov = prov.explode(col)

    # print("exploded prov")
    # print(prov)

    flat = (
        prov
        .stack()
        .dropna()
    )
    # print()
    # print("flat")
    # print()
    # print(flat)

    flat = flat.astype(float).astype(int)

    result_prov = (
        flat
        .groupby(level=0)
        .agg(set)
        .rename("_prov")
    )

    return df_with_many_prov_cols.drop(columns=prov_cols).join(result_prov)


def flatten(x):
    if isinstance(x, list):
        for item in x:
            yield from flatten(item)
    elif x is not None:
        yield x

def evaluate_provenance_fast(df):
    if isinstance(df, skrub.DataOp):
        df = df.skb.preview()

    prov_cols = [c for c in df.columns if c.startswith("_prov")]

    prov_series = (
        df[prov_cols]
        .apply(
            lambda row: {
                int(v)
                for cell in row
                for v in flatten(cell)
                if pd.notna(v)
            },
            axis=1
        )
        .rename("_prov")
    )

    return df.drop(columns=prov_cols).join(prov_series)

# region checklist of functions
# To adapt more functions take a look at:
# https://skrub-data.org/stable/reference/index.html+-