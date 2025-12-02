import pandas as pd
from importlib import reload
import skrub
from functools import wraps
from skrub import selectors as s
import importlib.util
import sys
from pathlib import Path


PROV_COLUMN = "_prov"

def EddieProvenanceReporter(func, kwargs, args):
    print(f"I know you executed {func.__name__}")

    print("now I will tell you the args")
    for argument in args:
            if type(argument) == list:
                print("-----------------------------------")
                print("check on the type of argument class list is successful")
                print("now iterating over each element of the list:")
                for i,elementi in enumerate(argument):
                    print(f"that is the {i}th element")
                    print(elementi)
            else:
                print("argument: ", argument)
                print("type(argument): ", type(argument))
                
    print("-----------------------------------")
    print("now I will go over keyword arguments")
    for k,v in kwargs.items():
            print("the keyword is: ", k, " the value will be printed below: ")
            print(v)

    return 0

def _product_provenances(p1: str, p2: str) -> str:
    """
    Combine two provenance expressions using semiring multiplication (*).
    """
    p1 = str(p1)
    p2 = str(p2)
    return f"({p1})*({p2})"
def _sum_provenances(p1: str, p2: str) -> str:
    """
    Combine two provenance expressions using semiring multiplication (*).
    """
    p1 = str(p1)
    p2 = str(p2)
    return f"({p1})+({p2})"

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
    EddieProvenanceReporter(_original_merge_top, kwargs, args)
    left = with_provenance(left, "left") # a check should be done if there are already provenance collumns, THIS IS NOT FUTURE PROOF
    right = with_provenance(right, "right")# A different apporach is wrapping the df class and running this funciton on creation of DF
    left = left.rename(columns={PROV_COLUMN: "_prov_left"}) # i did not do this as i did not figure out a sexy way to name the collumn
    right = right.rename(columns={PROV_COLUMN: "_prov_right"}) #in a generic way, where its called "prov_[df name]. That approach is deffinetly future proof and does not break"
    #will do with a switch later
    how = kwargs.get("how")
    if (how == "outer"): #OUTER JOIN IS AN UNION
        output = left.merge(right, on= kwargs.get("on"), how = "outer")
        provList = []
        for i in range(len(output["_prov_left"])):
            provList.append(f"{_sum_provenances(output['_prov_left'][i], output["_prov_right"][i])}")
        output[PROV_COLUMN] = provList
        output = output.drop(columns=["_prov_left", "_prov_right"])

        return output
    elif (how == "inner" or how == "left" or how == "right"): #INNER JOIN IS AN INTERSECTION
        # An inner join can be done as a cartesian product x selection for the provenance tuples
        # however is simpler and more logical if we just apply the rules for an intersection
        # also from my understanding, the inner join is a more concise case of left and right join
        # so the logic here for the polynomials should be the same
        output = left.merge(right, on= kwargs.get("on"), how = how)
        provList = []
        for i in range(len(output["_prov_left"])):
            provList.append(f"{_product_provenances(output['_prov_left'][i], output["_prov_right"][i])}")
        output[PROV_COLUMN] = provList
        output = output.drop(columns=["_prov_left", "_prov_right"])

        return output


# Patch top-level merge
pd.merge = provMerge

# Patch method on DataFrame



