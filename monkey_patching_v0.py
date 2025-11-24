import pandas as pd
from importlib import reload
import skrub
from functools import wraps
from skrub import selectors as s


df1 = pd.DataFrame({"Country": ["USA", "Italy", "Georgia"]})#.astype("string").set_index("Country")
df2 = pd.DataFrame({"Country": ["Spain", "Belgium", "Italy"],
                    "Capital": ["Madrid", "Brussel", "Rome"]})

def provenance_func_name(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
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
            print("the key is: ", k, " the value will be printed below: ")
            print(v)

        return func(*args,**kwargs)
    return wrapper

# To adapt more functions take a look at:
# https://skrub-data.org/stable/reference/index.html
dict_of_functions_with_implemented_provenance={
    "df.loc": pd.DataFrame.loc,                             # SELECT        -> Access a group of rows and columns by label(s) or a boolean array.
    "df.iloc": pd.DataFrame.iloc,                           # SELECT        -> apparently deprecated
    "series.loc": pd.Series.loc,                            # SELECT        -> Access a group of rows and columns by label(s) or a boolean array.
    "series.iloc": pd.Series.iloc,                          # SELECT        -> apparently deprecated
    "df.at": pd.DataFrame.at,                               # SELECT        -> access one value -> there is also series.at and df.iat, and series.iat
    "df[]": pd.DataFrame.__getitem__,                       # SELECT
    "skb.select" : skrub.DataOp.skb.select,                 # SELECT        -> over skb to select subset of columns
    "drop" : skrub.DataOp.skb.drop,                         # SELECT
    "assign": pd.DataFrame.assign,                          # PROJECT
    "concat" : skrub.DataOp.skb.concat,                     # somewhat join
    "get_data" : skrub.DataOp.skb.get_data,                 # idk
    "df.groupby" : pd.DataFrame.groupby,                    # AGGREGATE 
    "series.groupby": pd.Series.groupby,                    # AGGREGATE  
    "dfGroupBy.agg":pd.core.groupby.DataFrameGroupBy.agg,   # AGGREGATE 
    "seriesGroupBy.agg":pd.core.groupby.SeriesGroupBy.agg,  # AGGREGATE 
    "df.agg": pd.DataFrame.agg,                             # AGGREGATE 
    "series.agg": pd.Series.agg,                            # AGGREGATE 
    "agg": pd.core.groupby.DataFrameGroupBy.agg,            # AGGREGATE
    "pd.merge" : pd.merge,                                  # JOIN          -> example: pd.merge(df1,df2, on="Ids")
    "df.merge" : pd.DataFrame.merge,                        # JOIN          -> example: df1.merge(df2, on="Ids")
}


# bad practice of writting so many if-else statements -> asked gpt for improvement and he suggested code.. but I did not understand it yet
def wrap_skrub(func, names_of_original_functions, provenance_wrapper_for_the_function, verbose=False):
    @wraps(func)
    def wrapper(*args,**kwargs):
        
        for original_name, new_function in zip(names_of_original_functions, provenance_wrapper_for_the_function):
            if original_name in dict_of_functions_with_implemented_provenance.keys():
                # just understood that the following approach will not work -> will reassign the value in the dictionary
                # how can one make it more dynamic?
                # dict_of_functions_with_implemented_provenance[original_name] = new_function
                result_function = new_function(dict_of_functions_with_implemented_provenance[original_name])
                if original_name == "df[]":
                    pd.DataFrame.__getitem__ = result_function
                elif original_name == "df.loc":
                    pd.DataFrame.loc = result_function
                elif original_name == "df.iloc":
                    pd.DataFrame.iloc = result_function
                elif original_name == "series.loc":
                    pd.Series.loc = result_function
                elif original_name == "series.iloc":
                    pd.Series.iloc = result_function
                elif original_name == "df.at":
                    pd.DataFrame.at = result_function
                elif original_name == "pd.merge":
                    pd.merge = result_function            # downside: it decorates all pandas functions -> other skrub unrelated parts of the pipeline are also affected
                elif original_name == "df.merge":
                    pd.DataFrame.merge = result_function  # downside: it decorates all pandas functions -> other skrub unrelated parts of the pipeline are also affected
                elif original_name == "concat":
                    skrub.DataOp.skb.concat = result_function
                elif original_name == "get_data":
                    skrub.DataOp.skb.get_data = result_function
                elif original_name == "select":
                    skrub.DataOp.skb.select = result_function
                elif original_name == "drop":
                    skrub.DataOp.skb.drop = result_function
                elif original_name == "df.groupby":
                    pd.DataFrame.groupby = result_function
                elif original_name == "series.groupby":
                    pd.Series.groupby = result_function
                elif original_name == "agg":            # implement provenance for all 4-5 agg functions -> look into dict_of_functions_with_implemented_provenance 
                    pd.core.groupby.DataFrameGroupBy.agg = result_function
                elif original_name == "assign":
                    pd.DataFrame.assign = result_function
                else:
                    if verbose:
                        print("The value of the original function is in the dictionary, but not in the wrapper() function.")
                    
            else:
                if verbose:
                    print("One of the provenance functinos was not actiavated because no such key was found in dict_of_functions_with_implemented_provenance")
        return func(*args,**kwargs)
    return wrapper


