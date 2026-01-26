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

def dataop_key(a_dataop):
    return id(a_dataop._skrub_impl)         # was also thinking of id(a_dataop._skrub_impl.__dict__) -> chatgpt did not like it because "Unstable under refactoring; If skrub moves to: -> __slots__ -> or cached properties -> or C-extension-backed objects → this approach with __dict__ breaks immediately

DATAOPS_THAT_CHANGE_NOTHING = ["groupby"]


def export_hashdict_to_excel(hashdict, path="provenance_overview_debugging.xlsx"):
    index_rows = []

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for dataop_id, (df_name, df) in hashdict.items():
            sheet_name = f"df_{dataop_id}"

            # Excel sheet names have length limits
            sheet_name = sheet_name[:31]

            # Write DataFrame
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Collect index info
            index_rows.append({
                "dataop_id": dataop_id,
                "df_name": df_name,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": ", ".join(df.columns),
                "sheet_name": sheet_name
            })

        # Write overview sheet
        index_df = pd.DataFrame(index_rows)
        index_df.to_excel(writer, sheet_name="index", index=False)


import numpy as np

# region ProvenanceModule
class ProvenanceModule:
    # Pseudo code: hashdict = new hashdict() # a key is a hash and its value is a tuple (name_of_the_df, df_with_provenance_column) 
    def __init__(self):
        # key   -> dataop_key
        # value -> (df_name, pandas_df_with_provenance)
        # maybe there is a more optimized dictionary?
        self.hashdict: dict[int, pd.DataFrame] = {}         # TODO: consider using numpy arrays instead
    
    # def store_provenance_of_the_dataop(self, a_dataop, underlying_df):
    #     self.hashdict[dataop_key(a_dataop)] = underlying_df

    # def _get_df(self, a_dataop):
    #     dataop_id = dataop_key(a_dataop)
    #      # Cache hit
    #     if dataop_id in self.hashdict:
    #         underlying_df = self.hashdict[dataop_id]
    #         # print(f"[CACHE HIT] Using cached {df_name} DataFrame")
    #         return underlying_df
        
    #     # print(f"[CACHE MISS]")
    #     a_dataop_dictionary = get_dataop_dictionary(a_dataop=a_dataop)
    #     # print(a_dataop_dictionary)
    #     # region CHANGES NOTHING
    #     # List of DataOp names that only wrap the existing DataOps and do not change anything
    #     if "method_name" in a_dataop_dictionary and a_dataop_dictionary["method_name"] in DATAOPS_THAT_CHANGE_NOTHING:
    #         wrapped_dataop_id = dataop_key(a_dataop_dictionary["obj"])
    #         if wrapped_dataop_id in self.hashdict:
    #             # region TASK: copy()?
    #             underlying_df = self.hashdict[wrapped_dataop_id]
    #             # print(f"[CACHE SOMEWHAT HIT] Using cached {df_name} DataFrame")
    #             # print(f"""[CACHE] Storing provenance of {a_dataop_dictionary["method_name"]}""") 
    #             self.hashdict[dataop_id] = self.hashdict[wrapped_dataop_id] 
    #             return underlying_df
    #         else:
    #             # if the underlying dataop is a Var Dataop -> define it
    #             underlying_dataop_dict = get_dataop_dictionary(a_dataop_dictionary["obj"])
    #             if ("name" in underlying_dataop_dict) and ("value" in underlying_dataop_dict):
    #                 # print("################ hey it is a var dataop #############")
    #                 # TODO: wrap Var DataOp separately
    #                 # print( underlying_dataop_dict)
    #                 underlying_df = underlying_dataop_dict["name"], underlying_dataop_dict["value"]
    #                 self.hashdict[wrapped_dataop_id] = (df_name, with_provenance(underlying_df,df_name))
    #                 self.hashdict[dataop_id] = (df_name, with_provenance(underlying_df,df_name))
    #                 return (df_name, with_provenance(underlying_df,df_name))
    #             # # print("I know what is the problem!!")
    #             # # print(get_dataop_dictionary(a_dataop_dictionary["obj"]))
            
    #     if "name" in a_dataop_dictionary and "value" in a_dataop_dictionary:
    #         # TODO: get rid of this code by implementing coverage of Var DataOps!
    #         # print(f"[CACHE MISS] Defining provenance columns for a Var DataOp")
    #         # If a_dataop is a Var DataOp it contains the name and the df -> extracting them
    #         df_name = a_dataop_dictionary["name"]
    #         underlying_df = a_dataop_dictionary["value"]

    #         # Defining the PROV_COLUMN
    #         underlying_df = with_provenance(df=a_dataop_dictionary["value"], source_name=df_name)

    #         # Storing the dataframe with the provenance and its name
    #         #self.hashdict[dataop_id] = (df_name, underlying_df)
    #         self.store_provenance_of_the_dataop(a_dataop=a_dataop, underlying_df=underlying_df)

    #         return underlying_df
        
    #     # Neither cached nor a Var DataOp -> error
    #     raise RuntimeError("Provenance error: DataOp not found in cache and is not a Var DataOp."
    #                        "Traversing of the nested DataOps might be necessaary to implement"
    #                        f"The passed DataOp has this dictionary: {get_dataop_dictionary(a_dataop)}" )
    
    # region provenance_merge
    # def provenance_merge(self, a_dataop):
    #     # print("This thing gets executed")
    #     return a_dataop
        # pass # don't need to do anything
        
        ## print(self.hashdict)
        # # print("The passed dictionary is")
        # # print(resulting_dataop_dictionary)

    # region provenance_groupby
    def provenance_groupby(self,a_dataop):
        
        
        # groupby does not do anything
        ## print(resulting_dataop_dictionary["metadata"]["_creation_stack_lines"])
        # underlying_df = self._get_df(a_dataop)
        # region TASK:
        # print("[PROVENANCE] groupby start")
        # print(get_dataop_dictionary(a_dataop))
        # print(set_dataop_dictionary_val(a_dataop,"metadata_provenance",a_dataop.obj.columns))
        # print(get_dataop_dictionary(a_dataop))

        # print("[PROVENANCE] groupby end")


        # region implement for agrs and kwargs
        # return df_name, underlying_df
        return a_dataop

    # region override agg
    def provenance_agg(self,a_dataop):
        # resulting_dataop_dictionary =
        d =  get_dataop_dictionary(a_dataop)
        
        # print("dictionary before")
        # print(d)
        # print("#"*80)
        # print("dictionary of groupby inside")
        preview_of_dataop_inside = get_dataop_dictionary(d["obj"])["results"]
        # print(preview_of_dataop_inside)

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

        # Ensure canonical form
        # d["args"] = (agg_dict,)
        # d["kwargs"] = {}

        # ---- ADD _prov* columns (IN PLACE, FAST) ----
        # cols = list(d["obj"].obj.columns.copy())
        # print("""preview_of_dataop_inside["preview"]""")
        # print(preview_of_dataop_inside["preview"])
        object_inside_preview = preview_of_dataop_inside["preview"]
        if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
            cols = object_inside_preview.obj.columns
            groupby_keys = object_inside_preview._grouper.names
            # groupby_keys = set(object_inside_preview.obj.index.names)


            
        # elif isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
        elif isinstance(object_inside_preview, pd.DataFrame):
            # print("##############################")
            cols = object_inside_preview.columns

        # prov_cols = [c for c in cols if c.startswith("_prov")]
        # data_cols = [c for c in cols if not c.startswith("_prov")]

        # if isinstance(agg_dict, str) or callable(agg_dict):
        #     # apply to all columns
        #     agg_dict = {col: agg_dict for col in cols if not col.startswith("_prov") }

        # elif isinstance(agg_dict, list):
        #     agg_dict = {col: agg_dict for col in cols if not col.startswith("_prov") }

        if isinstance(agg_dict, str) or callable(agg_dict):
            # apply to all columns
            agg_dict = {col: agg_dict if not col.startswith("_prov") else list for col in cols } #1
            if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
                for k in groupby_keys:
                    agg_dict.pop(k, None)
            
            d["args"] = (agg_dict,)
            d["kwargs"] = {}
            # this function has a side effect -> it adjusts the dictionary in place, you do not need to 
            return a_dataop

        elif isinstance(agg_dict, list):
            agg_dict = {col: agg_dict if not col.startswith("_prov") else list for col in cols} #2
            if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
                for k in groupby_keys:
                    agg_dict.pop(k, None)
            
            d["args"] = (agg_dict,)
            d["kwargs"] = {}
            # this function has a side effect -> it adjusts the dictionary in place, you do not need to 
            return a_dataop
        
        elif isinstance(agg_dict, dict):
            # print("we enter this branch in the case of {'col_name':'max'} too")
            first_value = next(iter(agg_dict.values()), None)

            if isinstance(first_value, tuple):
                # print("agg_dict values are tuples (named aggregation style)")
                # example: {'new_col': ('old_col', 'max')}
                # or also (new_col_name = ('old_col', 'max'))
                for col in cols:
                    if col.startswith("_prov"):
                        agg_dict[col] = (col, list)
                d["kwargs"] = agg_dict
                return a_dataop       
            else:
                # example: {'col_name':'max'}
                for col in cols:
                    if col.startswith("_prov"):
                        agg_dict[col] = list
                
        #TODO: consider tuple instead of list # TODO: but it should be list, list is a transformation and cannot be mixed with aggregation functions due to pandas rules
        
        # print("agg_dict again!")
        # print(agg_dict)
        
        # print("dictionary after")
        # print(d)
        # earlier
        d["args"] = (agg_dict,)
        d["kwargs"] = {}
        # this function has a side effect -> it adjusts the dictionary in place, you do not need to 
        return a_dataop


    # left here just a reminder that the underlying idea is of the visit prototype from the python cookbook section 8.22
    def visit(self, node):
        methname = 'provenance_' + type(node).__name__
        meth = getattr(self, methname, None)
        if meth is None:
            meth = self.generic_visit
        return meth(node)
    def generic_visit(self, node):
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))
    

# Not sure if that is the right place to initialize provenance module and its hashdict
# As an alternative one can initialize it right after def enter_provenance_mode(func):
PROVENANCE_MODULE = ProvenanceModule()  


# def update_prov_column(df, transformer_name):
#     """
#     Returns a new DataFrame with an updated PROV_COLUMN:
#     new_value = transformer_name(old_value)

#     :param df: DataFrame containing PROV_COLUMN
#     :param transformer_name: e.g. dictionary["estimator"].__class__.__name__
#     :return: DataFrame with updated PROV_COLUMN
#     :rtype: pd.DataFrame
#     """
#     if PROV_COLUMN not in df.columns:
#         raise KeyError(f"'{PROV_COLUMN}' not found in DataFrame")

#     new_df = df.copy()

#     new_df[PROV_COLUMN] = (
#         transformer_name + "(" + new_df[PROV_COLUMN].astype(str) + ")"
#     )

#     return new_df

from monkey_patching_v02.data_provenance.provenance_utils_jeanne_performant import with_provenance_integers_shifted
import types
from sklearn.base import BaseEstimator, TransformerMixin, is_outlier_detector, is_classifier, is_regressor

from skrub import selectors as s
PROV_SELECTOR = s.filter_names(str.startswith, "_prov")

# region prov entry point
def enter_provenance_mode_dataop(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        
        result_dataop = None
        
        # print("[PROVENANCE]: Start")
        
        # print("printing result")
        # print(result)
        # print(type(result))
        # print("printing all arguments")
        for argument in args:
            # print("#"*40)
            # print("argument")
            # print(argument)
            # print("isinstance of dataop")
            # print(isinstance(argument, skrub.DataOp))
            # # print(isinstance(argument, skrub._data_ops._data_ops.DataOp))
            # # print(type(result_dataop) == skrub._data_ops._data_ops.DataOp)
            # # print(isinstance(argument, skrub._data_ops._data_ops.Var))
            # # print(type(argument)== skrub._data_ops._data_ops.Var)
            # # print("type of argument")
            # # print(type(argument))
            # print("#"*40)

            # print(argument)
            # if isinstance(argument, types.SimpleNamespace):
            #     print("#"*80)
            #     print("Found SimpleNamespace")
            #     print("#"*80)
            #     print(argument)
            #     result = func(*args,**kwargs)
            #     # print(type(result))
            #     result = with_provenance_integers_shifted(df=final_dict["value"], table_name=final_dict["name"])
            #     # print()
            #     # print("Result after adjusting SimpleNamespace")
            #     # print(result)
                
            #     print("[PROVENANCE VAR]: End")
            #     return result
            
            
            if isinstance(argument, skrub.DataOp):
                result_dataop = argument
                break
        # result_dataop = func(*args,**kwargs)


        # if type(result_dataop) == skrub._data_ops._data_ops.DataOp:                 # If that is a DataOp we can inspect what is stored inside
        if isinstance(result_dataop, skrub.DataOp):                 # If that is a DataOp we can inspect what is stored inside
            final_dict = get_dataop_dictionary(result_dataop)                         # Inspecting
            # if "name" in final_dict.keys() and "value" in final_dict.keys():
            #     print(">>> THIS IS Var DataOp")
            #     set_dataop_dictionary_val(a_dataop=result_dataop, 
            #                               attribute_name="value", # dataframe is stored under value -> we append the provenance column to it
            #                               new_val=with_provenance_integers_shifted(df=final_dict["value"], table_name=final_dict["name"]))
                
            # el
            if "method_name" in final_dict.keys():                                    # Having specific attribute in the dictionary classifies what kind of DataOp it is 
                # print(">>> THIS IS A CallMethod DataOp")
                # # ASPJ logic is covered here
                # # Pandas logic is covered here
                # print("    method_name =", final_dict["method_name"])
                # print("    obj =", final_dict.get("obj"))
                # print("    args =", final_dict.get("args"))
                # print("    kwargs =", final_dict.get("kwargs"))

                # print(f"""The applied method is {final_dict["method_name"]}""")
                corresponding_provenance_function = getattr(PROVENANCE_MODULE, "provenance_"+final_dict["method_name"], None)
                if corresponding_provenance_function is None:
                    # print(f"""[PROVENANCE] Can't find a provenance_{final_dict["method_name"]} in the ProvenanceModule.""")
                    pass
                else:
                    # print("corresponding_provenance_func is NOT None")
                    result_df = corresponding_provenance_function(result_dataop)

            elif "estimator" in final_dict:
                # print(">>> THIS IS An Apply DataOp")
                # print(argument)
                # print(final_dict)
                # print(final_dict.keys())
                # print(final_dict)
                # print("    name =", final_dict["name"])                
                # print("    X =", final_dict["X"])
                # print("    get_dataop_dictionary(X) =", get_dataop_dictionary(final_dict["X"]))
                # print("    y =", final_dict["y"])
                # print("    estimator =", final_dict["estimator"])

                est = final_dict["estimator"]
                if is_regressor(est) or is_classifier(est) or is_outlier_detector(est):
                    # print("one of them is XGB")
                    # print("this one: ", est)
                    # print(final_dict)
                    # print(final_dict.keys())
                    
                    dataop_X_dict = get_dataop_dictionary(final_dict["X"])

                    preview  = dataop_X_dict["results"]["preview"]
                    prov_cols = [col for col in preview.columns if col.startswith("_prov")]
                    # print("before final_dict['X']")
                    # print(final_dict["X"])
                    final_dict["X"] = final_dict["X"].drop(columns= prov_cols)
                    # print("after adjustment final_dict['X']")
                    # print(final_dict["X"])
                    
                    # print("maybe there is something else in dataopX_dict")
                    # print("#"*80)
                    # print(dataop_X_dict)
                    # print("#"*80)

                    # print("maybe preview is not used?")
                    # print(preview)
                    # prov_cols = [col for col in preview.columns if col.startswith("_prov")]
                    # for pcol in prov_cols:
                    #     print(pcol)
                    # X_prov = s.select(preview, PROV_SELECTOR)
                    # X_main = s.select(preview, final_dict["cols"] - PROV_SELECTOR)
                    # print("#"*80)
                    # print("Apparently this dataframe contains lists of lists")
                    
                    # # mutate preview before estimator runs
                    # dataop_X_dict["results"]["preview"] = X_main
                    # dataop_X_dict["results"]["value"] = X_main
                    
                    result = func(*args, **kwargs)


                    # X_out = pd.concat([result, X_prov], axis=1)

                    # # update Apply DataOp result (this is the critical part for propagation)
                    # set_dataop_dictionary_val(
                    #     a_dataop=result_dataop,
                    #     attribute_name="results",
                    #     new_val={
                    #         **final_dict["results"],
                    #         "value": result,
                    #         "preview": result,
                    #     }
                    # )

                    return result
                elif isinstance(est, skrub._select_cols.SelectCols):                    
                    final_dict["estimator"].__dict__["cols"] = final_dict["estimator"].__dict__["cols"] | PROV_SELECTOR

                    # Only the dictionary of the estimator plays a role -> not of the dataop.
                    # set_dataop_dictionary_val(a_dataop=result_dataop,
                    #                           attribute_name="cols",
                    #                           new_val=final_dict["cols"] | PROV_SELECTOR)
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


# region enable_provenance
def enable_why_data_provenance():
    set_provenance(skrub._data_ops._evaluation,"evaluate", provenance_func=enter_provenance_mode_dataop)
    set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)


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


from collections.abc import Iterable

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

from provenance_utils_jeanne_performant import decode_prov

def decode_prov_column(df, evaluate_provenance_first=True):
    """
    Decode 64-bit integer provenance IDs into a human-readable format.

    This function transforms the values in the ``"_prov"`` column from
    encoded 64-bit integers into the form::

        table_name:row_id

    It should be applied **after** provenance columns have been evaluated
    and consolidated into a single ``"_prov"`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing a ``"_prov"`` column with encoded provenance IDs.

    evaluate_provenance_first : bool
        If True, all ``_prov*`` columns are evaluated and consolidated into a
        single ``"_prov"`` column prior to decoding. If False, the function
        assumes that a ``"_prov"`` column already exists.
        
    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with decoded, more interpretable
        provenance identifiers.
    """
    new_df = df.copy()


    if evaluate_provenance_first:
        new_df = evaluate_provenance_fast(new_df) 

    new_df["_prov"] = new_df["_prov"].map(lambda set_x: [decode_prov(x) for x in set_x])
    return new_df

# region checklist of functions
# To adapt more functions take a look at:
# https://skrub-data.org/stable/reference/index.html+-