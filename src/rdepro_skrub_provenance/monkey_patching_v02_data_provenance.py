from __future__ import annotations
import pandas as pd
import skrub
from skrub import selectors as s
from functools import wraps
from sklearn.base import is_outlier_detector, is_classifier, is_regressor
from rdepro_skrub_provenance.provenance_utils import decode_prov, with_provenance_integers_shifted



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
    def __init__(self):
        self.hashdict: dict[int, pd.DataFrame] = {}   

    def provenance_groupby(self,a_dataop):
        return a_dataop

    def provenance_agg(self,a_dataop):
        d =  get_dataop_dictionary(a_dataop)
        preview_of_dataop_inside = get_dataop_dictionary(d["obj"])["results"]
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

        if isinstance(agg_dict, str) or callable(agg_dict):
            agg_dict = {col: agg_dict if not col.startswith("_prov") else list for col in cols } #1
            if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
                for k in groupby_keys:
                    agg_dict.pop(k, None)
            
            d["args"] = (agg_dict,)
            d["kwargs"] = {}
            return a_dataop

        elif isinstance(agg_dict, list):
            agg_dict = {col: agg_dict if not col.startswith("_prov") else list for col in cols} #2
            if isinstance(object_inside_preview, pd.core.groupby.generic.DataFrameGroupBy):
                for k in groupby_keys:
                    agg_dict.pop(k, None)
            
            d["args"] = (agg_dict,)
            d["kwargs"] = {}
            return a_dataop
        
        elif isinstance(agg_dict, dict):
            first_value = next(iter(agg_dict.values()), None)

            if isinstance(first_value, tuple):
                for col in cols:
                    if col.startswith("_prov"):
                        agg_dict[col] = (col, list)
                d["kwargs"] = agg_dict
                return a_dataop       
            else:
                for col in cols:
                    if col.startswith("_prov"):
                        agg_dict[col] = list
                
        d["args"] = (agg_dict,)
        d["kwargs"] = {}
        return a_dataop

    def visit(self, node):
        methname = 'provenance_' + type(node).__name__
        meth = getattr(self, methname, None)
        if meth is None:
            meth = self.generic_visit
        return meth(node)
    def generic_visit(self, node):
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))
    

PROVENANCE_MODULE = ProvenanceModule()  

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
            # If that is a DataOp we can inspect what is stored inside
        if isinstance(result_dataop, skrub.DataOp):                 # If that is a DataOp we can inspect what is stored inside
            final_dict = get_dataop_dictionary(result_dataop)                         # Inspecting
     
            if "method_name" in final_dict.keys():                                    # Having specific attribute in the dictionary classifies what kind of DataOp it is 

                corresponding_provenance_function = getattr(PROVENANCE_MODULE, "provenance_"+final_dict["method_name"], None)
                if corresponding_provenance_function is None:
                    # print(f"""[PROVENANCE] Can't find a provenance_{final_dict["method_name"]} in the ProvenanceModule.""")
                    pass
                else:
                    # print("corresponding_provenance_func is NOT None")
                    result_df = corresponding_provenance_function(result_dataop)

            elif "estimator" in final_dict:
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
            
            elif "attr_name" in final_dict:
                # TODO: introduce provenance, if one column is selected -> attach to it all prov_cols -> risky if for example ApplyToCols takes one column and gets a dataframe..
                pass
            else:
                pass

        return func(*args, **kwargs)
    return wrapper

def enter_provenance_mode_var(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        
        for argument in args:
 
            if isinstance(argument, skrub._data_ops._data_ops.Var):
  
                final_dict = get_var_dictionary(argument)

        
        result = func(*args,**kwargs)


        if isinstance(result, pd.DataFrame):
            result = with_provenance_integers_shifted(
                df=result,
                table_name=final_dict["name"]
            )
        

        return result
    return wrapper

def set_provenance(namespace, name_of_the_function, provenance_func=enter_provenance_mode_dataop):
    current = getattr(namespace, name_of_the_function, None)
    if current is None:
        raise AttributeError(f"{namespace}.{name_of_the_function} not found")
    if getattr(current, "__provenance_patch__", False):
        return
    wrapped = provenance_func(current)
    setattr(wrapped, "__provenance_patch__", True)
    setattr(namespace, name_of_the_function, wrapped)




# region enable_provenance
def enable_why_data_provenance():
    set_provenance(skrub._data_ops._evaluation,"evaluate", provenance_func=enter_provenance_mode_dataop)
    set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)



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


    flat = (
        prov
        .stack()
        .dropna()
    )

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

def _is_prov_col(col) -> bool:
    if isinstance(col, str):
        return col.startswith("_prov")
    if isinstance(col, tuple) and len(col) > 0 and isinstance(col[0], str):
        return col[0].startswith("_prov")
    return False

def _make_prov_column(df, prov_series):
 
    if df.columns.nlevels == 1:
        return df.join(prov_series)
    prov_df = prov_series.to_frame()
    new_cols = pd.MultiIndex.from_tuples(
        [("_prov",) + ("",) * (df.columns.nlevels - 1)]
    )
    prov_df.columns = new_cols

    return pd.concat([df, prov_df], axis=1)



def evaluate_provenance_fast(df):
    if isinstance(df, skrub.DataOp):
        df = df.skb.preview()

    prov_cols = [c for c in df.columns if _is_prov_col(c)]

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

    base = df.drop(columns=prov_cols)
    return _make_prov_column(base, prov_series)



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
