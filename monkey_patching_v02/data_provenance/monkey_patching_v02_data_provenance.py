from __future__ import annotations
import pandas as pd
import skrub
from functools import wraps
# from provenance_utils_jeanne_performant import merge_with_provenance, groupby_aggregate_with_provenance, with_provenance
# from provenance_utils_jeanne_performant import PROV_COLUMN


# region Helpers
def get_dataop_dictionary(a_dataop):
    return a_dataop.__dict__

def set_dataop_dictionary_val(a_dataop, attribute_name, new_val):
    a_dataop.__dict__[attribute_name] = new_val

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
        # print("hey")
        d =  a_dataop.__dict__
        preview_of_dataop_inside = d["obj"].__dict__["_skrub_impl"].__dict__["results"]

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

        # ---- ADD _prov* columns (IN PLACE, FAST) ----
        if "preview" in preview_of_dataop_inside:
            object_inside_preview = preview_of_dataop_inside["preview"]
        else:
            object_inside_preview = preview_of_dataop_inside["fit_transform"]
        if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
            cols = object_inside_preview.obj.columns
            groupby_keys = object_inside_preview._grouper.names
            # groupby_keys = set(object_inside_preview.obj.index.names)


            
        # elif isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
        elif isinstance(object_inside_preview, pd.DataFrame):
            # print("##############################")
            cols = object_inside_preview.columns
            groupby_keys =[]


        if isinstance(agg_dict, str) or callable(agg_dict):
            # apply to all columns
            agg_dict = {col: agg_dict for col in cols if not col.startswith("_prov") }

        elif isinstance(agg_dict, list):
            agg_dict = {col: agg_dict for col in cols if not col.startswith("_prov") }

        # Remove groupby keys from aggregation dict
        for k in groupby_keys:
            agg_dict.pop(k, None)

        for c in cols:
            # startswith is faster than regex
            if c.startswith("_prov"):
                agg_dict[c] = list      #TODO: consider tuple instead of list # TODO: but it should be list, list is a transformation and cannot be mixed with aggregation functions due to pandas rules

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
from pipelines.monkey_patching_v02.data_provenance.provenance_utils_jeanne_performant import with_provenance_integers_shifted
# from pipelines.monkey_patching_v02.data_provenance.provenance_utils_jeanne_performant import with_provenance_integers_shifted
import types
from sklearn.base import BaseEstimator, TransformerMixin, is_outlier_detector, is_classifier, is_regressor

from skrub import selectors as s
PROV_SELECTOR = s.filter_names(str.startswith, "_prov")

# region prov entry point
def enter_provenance_mode_dataop_callmethod(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        # print("args")
        # print(args)
        the_callmethod_dataop = args[0]
        print("[PROVENANCE] Start ")
        print(the_callmethod_dataop)
        # print(isinstance(the_callmethod_dataop, skrub._data_ops._data_ops.Apply))
        # print(the_callmethod_dataop.estimator)
        if "method_name" in the_callmethod_dataop.__dict__ and the_callmethod_dataop.method_name != "agg":
            print("[PROVENANCE] End ", the_callmethod_dataop.method_name)

            return func(*args,**kwargs)
        elif "method_name" in the_callmethod_dataop.__dict__:
            PROVENANCE_MODULE.provenance_agg(the_callmethod_dataop)
            # print("[PROVENANCE] End ", the_callmethod_dataop.method_name)

            # print()
            return func(*args, **kwargs) # Just execute the function and get the result

    return wrapper


def enter_provenance_mode_apply_format_predictions(func):
    from functools import wraps
    import pandas as pd

    @wraps(func)
    def wrapper(self, X, pred):
        # Call original formatter
        out = func(self, X, pred)

        # Only attach provenance if output is tabular
        if not isinstance(out, (pd.DataFrame, pd.Series)):
            return out

        X_prov = s.select(X, PROV_SELECTOR)
        if X_prov is None or len(X_prov.columns) == 0:
            return out

        if isinstance(out, pd.Series):
            out = out.to_frame(name=out.name or "prediction")

        return pd.concat([out, X_prov], axis=1)

    return wrapper






def enter_provenance_mode_dataop_apply_x(func):
    from functools import wraps

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        apply_dataop = args[0]
        final_dict = apply_dataop.__dict__
        est = apply_dataop.estimator

        # Wrap estimator predict (or transform) if it's a model
        if is_regressor(est) or is_classifier(est) or is_outlier_detector(est):
            orig_predict = est.predict
            print("hey")
            print("apply_dataop.X")
            print(apply_dataop.X)
            print("its type")
            print(type(apply_dataop.X))
            print("maybe its preview")
            # print(apply_dataop.X.preview)
            # Split provenance columns from main data
            X_prov = s.select(apply_dataop.X, PROV_SELECTOR)
            X_main = s.select(apply_dataop.X, final_dict["cols"] - PROV_SELECTOR)

            # Run original prediction
            y = func(X_main, *args[1:], **kwargs)

            # Convert Series -> DataFrame if needed
            if isinstance(y, pd.Series):
                y = y.to_frame(name=y.name or "prediction")

            # Concatenate provenance columns
            return pd.concat([y, X_prov], axis=1)

        # Call the original apply() function
        return func(*args, **kwargs)

    return wrapper_func





        #         est = final_dict["estimator"]
        #         if is_regressor(est) or is_classifier(est) or is_outlier_detector(est):

        #             dataop_X_dict = get_dataop_dictionary(final_dict["X"])
        #             preview  = dataop_X_dict["results"]["preview"]

        #             X_prov = s.select(preview, PROV_SELECTOR)
        #             X_main = s.select(preview, final_dict["cols"] - PROV_SELECTOR)

        #             # mutate preview before estimator runs
        #             dataop_X_dict["results"]["preview"] = X_main

        #             result = func(*args, **kwargs)

        #             X_out = pd.concat([result, X_prov], axis=1)

        #             # update Apply DataOp result (this is the critical part for propagation)
        #             set_dataop_dictionary_val(
        #                 a_dataop=result_dataop,
        #                 attribute_name="results",
        #                 new_val={
        #                     **final_dict["results"],
        #                     "value": X_out,
        #                     "preview": X_out,
        #                 }
        #             )

        #             return X_out
        #         else:
        #             set_dataop_dictionary_val(a_dataop=result_dataop,
        #                                       attribute_name="cols",
        #                                       new_val=final_dict["cols"] - PROV_SELECTOR)
    return wrapper_func

def enter_provenance_mode_dataop_callmethod_gpt(func):
    """
    Wrap CallMethod.compute to inject provenance tracking for 'agg'.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        callmethod_node = args[0]  # The CallMethod instance

        method_name = getattr(callmethod_node, "method_name", None)
        print(f"[PROVENANCE] Start {method_name}")

        # Only intercept 'agg'
        if method_name == "agg":
            # Inject provenance: call your provenance aggregation routine
            # This should modify the preview / results of the DataOp before actual computation
            try:
                PROVENANCE_MODULE.provenance_agg(callmethod_node)
                # Optionally attach an empty _prov list to the preview so it exists
                preview = getattr(callmethod_node, "_preview", None)
                if preview is not None:
                    preview["_prov"] = []  # Initialize provenance list
            except Exception as e:
                print(f"[PROVENANCE] Failed during provenance injection: {e}")

        # Execute the actual method to get the result
        result = func(*args, **kwargs)

        print(f"[PROVENANCE] End {method_name}")
        return result

    return wrapper
        # print("inside isinstance(result_dataop, skrub.DataOp)")

        # if "method_name" in final_dict.keys():                                    # Having specific attribute in the dictionary classifies what kind of DataOp it is 
        #     print("checking the corresponding provenance function")
        #     corresponding_provenance_function = getattr(PROVENANCE_MODULE, "provenance_"+final_dict["method_name"], None)
        #     if corresponding_provenance_function is None:
        #         # print(f"""[PROVENANCE] Can't find a provenance_{final_dict["method_name"]} in the ProvenanceModule.""")
        #         pass
        #     else:
        #         print("corresponding_provenance_func is NOT None")
        #         corresponding_provenance_function(result_dataop)

        # TODO: continue witht the apply dataop here
        # if isinstance(result_dataop, skrub.DataOp):                 # If that is a DataOp we can inspect what is stored inside
            

        #     if "estimator" in final_dict:


        #         est = final_dict["estimator"]
        #         if is_regressor(est) or is_classifier(est) or is_outlier_detector(est):

        #             dataop_X_dict = get_dataop_dictionary(final_dict["X"])
        #             preview  = dataop_X_dict["results"]["preview"]

        #             X_prov = s.select(preview, PROV_SELECTOR)
        #             X_main = s.select(preview, final_dict["cols"] - PROV_SELECTOR)

        #             # mutate preview before estimator runs
        #             dataop_X_dict["results"]["preview"] = X_main

        #             result = func(*args, **kwargs)

        #             X_out = pd.concat([result, X_prov], axis=1)

        #             # update Apply DataOp result (this is the critical part for propagation)
        #             set_dataop_dictionary_val(
        #                 a_dataop=result_dataop,
        #                 attribute_name="results",
        #                 new_val={
        #                     **final_dict["results"],
        #                     "value": X_out,
        #                     "preview": X_out,
        #                 }
        #             )

        #             return X_out
        #         else:
        #             set_dataop_dictionary_val(a_dataop=result_dataop,
        #                                       attribute_name="cols",
        #                                       new_val=final_dict["cols"] - PROV_SELECTOR)
                

                
        #         # # print(" type of estimator =", type(final_dict["estimator"]))
        #     elif "attr_name" in final_dict:
        #         # TODO: introduce provenance, if one column is selected -> attach to it all prov_cols -> risky if for example ApplyToCols takes one column and gets a dataframe..
        #         pass

        #     else:
        #         pass

            
        # print("[PROVENANCE]: END")
            ## print(final_dict)

    #     return func(*args, **kwargs) # Just execute the function and get the result
    # return wrapper

# region provenance var

def enter_provenance_mode_var(func):
    
    @wraps(func)
    def wrapper(*args,**kwargs):
        # print("[PROVENANCE var] Start")

        # print("[PROVENANCE var] the args are")
        # print(args)
        # for i, argument in enumerate(args):
        #     print(f"{i}:{argument}")
            
        
        # result_dataop = None
        
        # print("[PROVENANCE Var]: Start")
        
        # print("printing result")
        # print(result)
        # print(type(result))
        # print("printing all arguments")
        # for argument in args:
        #     # print("#"*40)
        #     # print("argument")
        #     # print(argument)
        #     # # print("isinstance of dataop")
        #     # # print(isinstance(argument, skrub.DataOp))
        #     # # print(isinstance(argument, skrub._data_ops._data_ops.DataOp))
        #     # # print(type(result_dataop) == skrub._data_ops._data_ops.DataOp)
        #     # # print(isinstance(argument, skrub._data_ops._data_ops.Var))
        #     # # print(type(argument)== skrub._data_ops._data_ops.Var)
        #     # print("type of argument")
        #     # print(type(argument))
        #     # print("#"*40)
        #     if isinstance(argument, skrub._data_ops._data_ops.Var):
        #         # print("I guess the CallMethod has var")
        #         # print("check worked!")
        #         # skrub.Var is the _skrub_impl -> to get the dictionary just call argument.__dict__
        #         # Note it uses get_var_dictionary, not get_dataop_dictionary 
        #         final_dict = get_var_dictionary(argument)
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
                
        
        # result = func(*args,**kwargs)
        # print(type(result))
        # final_dict = get_var_dictionary(args[0])

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
        # print("[PROVENANCE Var]: Start")
        # print(result)
        # print("[PROVENANCE VAR]: End")

        return with_provenance_integers_shifted(df=args[0].__dict__["value"], table_name=args[0].__dict__["name"]) # Just execute the function and get the result
    return wrapper

def enter_provenance_mode_var(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        return with_provenance_integers_shifted(df=args[0].__dict__["value"], table_name=args[0].__dict__["name"]) # Just execute the function and get the result
    return wrapper
def set_provenance(namespace, name_of_the_function, provenance_func=enter_provenance_mode_dataop_callmethod):
    skrub_eval_namespace = namespace
    name = name_of_the_function
    skrub_eval = getattr(skrub_eval_namespace,name,None)
    setattr(skrub_eval_namespace, name, provenance_func(skrub_eval))
    # print(f"Set provenance for {name}")


from skrub._data_ops._evaluation import _Evaluator
from skrub._data_ops._data_ops import CallMethod


def enter_provenance_mode_evaluator_eval_data_op(orig_func):
    @wraps(orig_func)
    def wrapper(self, data_op):
        impl = data_op._skrub_impl

        # Inject BEFORE preview is computed
        if (isinstance(impl, CallMethod) and impl.method_name == "agg" and self.mode == "preview"):
            args = impl.args or ()
            if args and isinstance(args[0], dict) and "_prov" not in args[0]: # do you need it here? -> and "_prov" not in args[0] 
                PROVENANCE_MODULE.provenance_agg(impl)

        # IMPORTANT: capture and return the value
        result = yield from orig_func(self, data_op)
        return result
    pd.DataFrame.agg

    return wrapper


def set_provenance_evaluator(cls, method_name, provenance_wrapper):
    """
    Monkey-patch an Evaluator method with provenance support.
    """
    orig = getattr(cls, method_name)

    # Avoid double patching
    if getattr(orig, "_is_provenance_wrapped", False):
        return

    wrapped = provenance_wrapper(orig)
    wrapped._is_provenance_wrapped = True

    setattr(cls, method_name, wrapped)


# region checklist of functions
# To adapt more functions take a look at:
# https://skrub-data.org/stable/reference/index.html+-


