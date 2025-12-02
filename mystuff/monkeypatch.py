import pandas as pd
from importlib import reload
import skrub
from functools import wraps
from skrub import selectors as s
import importlib.util
import sys
from pathlib import Path


PROV_COLUMN = "_prov"



def _product_provenances(p1: str, p2: str) -> str:
    """
    Combine two provenance expressions using semiring multiplication (*).
    """
    p1 = str(p1)
    p2 = str(p2)
    return f"({p1})*({p2})"

def with_provenance(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Initialize provenance for a base relation.

    Each row receives a unique provenance identifier derived from the
    source table name and the row index. This corresponds to the leaf
    initialization rule in Perm (R0 in spirit): provenance initialization
    at the leaves of the data pipeline.
    """
    df = df.copy()
    df[PROV_COLUMN] = [f"{source_name}:{i}" for i in range(len(df))]
    return df

# wrap pd.merge, the function of the calss has to wrapped on its

_original_merge_top = pd.merge

@wraps(_original_merge_top)
def provMerge(left: pd.DataFrame, right: pd.DataFrame, *args, **kwargs):
    left = with_provenance(left, "left") 
    right = with_provenance(right, "right")
    left = left.rename(columns={PROV_COLUMN: "_prov_left"})
    right = right.rename(columns={PROV_COLUMN: "_prov_right"})
    #will do with a switch later
    if (kwargs["how"] == "outer"):  #a full outer join is just a cartesian product, here I use the logic from the cartesian function
        left["_tmp_key"] = 1        # in Jeanne's code
        right["_tmp_key"] = 1
        output = left.merge(right, on= "_tmp_key").drop(columns="_tmp_key")
        provList = []
        for i in range(len(output["_prov_left"])):
            provList.append(f"{_product_provenances(output['_prov_left'][i], output["_prov_right"][i])}")
        output[PROV_COLUMN] = provList
        output = output.drop(columns=["_prov_left", "_prov_right"])

        return output



# Patch top-level merge
pd.merge = provMerge

# Patch method on DataFrame

#@wraps(pd.DataFrame.merge)
#def leftRightJoin(*args, **kwags):


