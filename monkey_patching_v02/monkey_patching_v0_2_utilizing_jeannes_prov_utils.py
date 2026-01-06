from __future__ import annotations
import pandas as pd
import skrub
from functools import wraps
from pipelines.monkey_patching_v02.provenance_utils_jeanne import merge_with_provenance, groupby_aggregate_with_provenance, with_provenance

# region Helpers
def get_dataop_dictionary(a_dataop):
    return a_dataop._skrub_impl.__dict__

def dataop_key(a_dataop):
    return id(a_dataop._skrub_impl)         # was also thinking of id(a_dataop._skrub_impl.__dict__) -> chatgpt did not like it because "Unstable under refactoring; If skrub moves to: -> __slots__ -> or cached properties -> or C-extension-backed objects → this approach with __dict__ breaks immediately

DATAOPS_THAT_CHANGE_NOTHING = ["groupby"]


def export_hashdict_to_excel(hashdict, path="provenance_overview.xlsx"):
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
        self.hashdict: dict[int, tuple[str, pd.DataFrame]] = {}
    
    def store_provenance_of_the_dataop(self, a_dataop, df_name, underlying_df):
        dataop_id = dataop_key(a_dataop)
        self.hashdict[dataop_id] = (df_name, underlying_df)

    def _get_df(self, a_dataop):
        dataop_id = dataop_key(a_dataop)
         # Cache hit
        if dataop_id in self.hashdict:
            df_name, underlying_df = self.hashdict[dataop_id]
            print(f"[CACHE HIT] Using cached {df_name} DataFrame")
            return df_name, underlying_df
        
        print(f"[CACHE MISS]")
        a_dataop_dictionary = get_dataop_dictionary(a_dataop=a_dataop)
        # region CHANGES NOTHING
        # List of DataOp names that only wrap the existing DataOps and do not change anything
        if "method_name" in a_dataop_dictionary and a_dataop_dictionary["method_name"] in DATAOPS_THAT_CHANGE_NOTHING:
            wrapped_dataop_id = dataop_key(a_dataop_dictionary["obj"])
            if wrapped_dataop_id in self.hashdict:
                # region TASK: copy()?
                df_name, underlying_df = self.hashdict[wrapped_dataop_id]
                print(f"[CACHE SOMEWHAT HIT] Using cached {df_name} DataFrame")
                print(f"""[CACHE] Storing provenance of {a_dataop_dictionary["method_name"]}""") 
                self.hashdict[dataop_id] = self.hashdict[wrapped_dataop_id] 
                return df_name, underlying_df
            
        if "name" in a_dataop_dictionary and "value" in a_dataop_dictionary:
            print(f"[CACHE MISS] Defining provenance columns for a Var DataOp")
            # If a_dataop is a Var DataOp it contains the name and the df -> extracting them
            df_name = a_dataop_dictionary["name"]
            underlying_df = a_dataop_dictionary["value"]

            # Defining the PROV_COLUMN
            underlying_df = with_provenance(df=a_dataop_dictionary["value"], source_name=df_name)

            # Storing the dataframe with the provenance and its name
            #self.hashdict[dataop_id] = (df_name, underlying_df)
            self.store_provenance_of_the_dataop(a_dataop=a_dataop, df_name=df_name, underlying_df=underlying_df)

            return df_name, underlying_df
        
        # Neither cached nor a Var DataOp -> error
        raise RuntimeError("Provenance error: DataOp not found in cache and is not a Var DataOp."
                           "Traversing of the nested DataOps might be necessaary to implement"
                           f"The passed DataOp has this dictionary: {get_dataop_dictionary(a_dataop)}" )
        
    def print_and_return_name_and_its_pandas_df(self, a_dataop, str_name):
        a_dataop_dictionary = a_dataop._skrub_impl.__dict__
        print(f"The name of the {str_name} dataop is: ", a_dataop_dictionary["name"])
        print(f"""The underlying {type(a_dataop_dictionary["value"])} is: """)
        print(a_dataop_dictionary["value"])
        return a_dataop_dictionary["name"], a_dataop_dictionary["value"]
    
    # region provenance_merge
    def provenance_merge(self, a_dataop):
        resulting_dataop_dictionary = get_dataop_dictionary(a_dataop)
        assert resulting_dataop_dictionary["method_name"] == "merge", "Are you sure you want to apply provenance_merge not on merge operation? comment this out then."
        left_dataop = resulting_dataop_dictionary["obj"]
        print("Left DataOp is:")
        print(left_dataop)
        # Pseudo code: if hash(left_dataop) in hashdict.keys: name_df_left, df_left = hashdict[hash(left_dataop)] else do the next line
        
        name_df_left, df_left = self._get_df(left_dataop)  # self.print_and_return_name_and_its_pandas_df(left_dataop, "left")

        print("----------------------")
        right_dataop = None
        for arg in resulting_dataop_dictionary["args"]:
            if type(arg) == skrub._data_ops._data_ops.DataOp:
                right_dataop = arg
                print("Right DataOp is:")
                print(right_dataop)
                # Pseudo code: if hash(right_dataop) in hashdict.keys: name_df_right, df_right = hashdict[hash(right_dataop)] else do the next line
                name_df_right, df_right = self._get_df(right_dataop) # self.print_and_return_name_and_its_pandas_df(right_dataop, "right")

        # region TASK:
        #
        #
        #
        #
        #
        # region support args too
        print("----------------------")
        print("Now we go over keyword arguments.")
        for k, w in resulting_dataop_dictionary["kwargs"].items():
            if k == "how":
                how_merge = w
             # region support left_on, right_on
            if k == "on":                
                on_merge = w
            print(f"{k}:{w}")
        print("[PROVENANCE] Final result:")
        final_result = merge_with_provenance(left=df_left,
                                    left_source_name=name_df_left,
                                    right=df_right,
                                    right_source_name=name_df_right,
                                    how=how_merge,
                                    on=on_merge)
        print(final_result)
        print("[PROVENANCE Merge] Executed provenance_merge().")
        """ moved it to the provenance entrance
        print("The provenance hashdict follows:")
        for k,w in self.hashdict.items():
            print(f"{k}: {w}")
        """
        name_of_this_execution = f"[merge] ({name_df_left}) AS left_dataop {how_merge} JOIN ({name_df_right}) AS right_dataop ON {on_merge}"
        return name_of_this_execution, final_result
        #print(self.hashdict)
        # print("The passed dictionary is")
        # print(resulting_dataop_dictionary)

    # region provenance_groupby
    def provenance_groupby(self,a_dataop):
        
        
        # groupby does not do anything
        #print(resulting_dataop_dictionary["metadata"]["_creation_stack_lines"])
        df_name, underlying_df = self._get_df(a_dataop)
        # region TASK:





        # region implement for agrs and kwargs
        return f"groupby on top of that [{df_name}]", underlying_df
    
    """ old version
    # region provenance_agg
    # More detailed printing used in the development
    def provenance_agg(self,resulting_dataop_dictionary):
        print("the agg dictionary is")
        print(resulting_dataop_dictionary)
        print("#### End agg dictionary ###")
        group_by_dataop = resulting_dataop_dictionary["obj"]
        print("the groupby dictionary is")
        print(get_dataop_dictionary(group_by_dataop))
        print("#### End groupby dictionary ###")
        var_dataop = get_dataop_dictionary(group_by_dataop)["obj"]

        underlying_df = get_dataop_dictionary(var_dataop)["value"]
        underlying_df_name =  get_dataop_dictionary(var_dataop)["name"]
        # region TASK:
        # 
        # 
        # 
        # 
        # 
        #  
        # region include kwargs
        agg_operator = resulting_dataop_dictionary["args"][0]
        print("Aggregation Operator is: ", agg_operator)
        aggregated_on = get_dataop_dictionary(group_by_dataop)["args"][0]
        print("Aggregating on: ", aggregated_on)
        print("The dataframe being aggregated is the following")
        print(underlying_df)
        print()
        result_agg_with_prov = groupby_aggregate_with_provenance(df=with_provenance(df=underlying_df,
                                                                                    source_name=underlying_df_name),
                                                                 by=aggregated_on,
                                                                 agg_spec=agg_operator)
        print(result_agg_with_prov)
        print("[PROVENANCE FOR AGG] End for this operator.")
        """
    # region override agg
    def provenance_agg(self,a_dataop):
        resulting_dataop_dictionary = get_dataop_dictionary(a_dataop)
        print("the agg dictionary is")
        print(resulting_dataop_dictionary)
        print("#### End agg dictionary ###")
        group_by_dataop = resulting_dataop_dictionary["obj"]
        print("the groupby dictionary is")
        print(get_dataop_dictionary(group_by_dataop))
        print("#### End groupby dictionary ###")
        var_dataop = get_dataop_dictionary(group_by_dataop)["obj"]
        
        # testing not sure how to connect everything
        underlying_df_name, underlying_df = self._get_df(group_by_dataop)
        #underlying_df = get_dataop_dictionary(var_dataop)["value"]
        #underlying_df_name =  get_dataop_dictionary(var_dataop)["name"]

        # region TASK:
        # 
        # 
        # 
        # 
        # 
        #  
        # region include kwargs
        agg_operator = resulting_dataop_dictionary["args"][0]
        print("Aggregation Operator is: ", agg_operator)
        aggregated_on = get_dataop_dictionary(group_by_dataop)["args"][0]
        print("Aggregating on: ", aggregated_on)
        print("The dataframe being aggregated is the following")
        print(underlying_df)
        print()
        result_agg_with_prov = groupby_aggregate_with_provenance(df=with_provenance(df=underlying_df,
                                                                                    source_name=underlying_df_name),
                                                                 by=aggregated_on,
                                                                 agg_spec=agg_operator)
        print(result_agg_with_prov)
        print("[PROVENANCE FOR AGG] End for this operator.")
        name_of_this_operator = f"[aggregate] SELECT {agg_operator}(*) FROM ({underlying_df_name}) GROUP BY {aggregated_on} "
        return name_of_this_operator, result_agg_with_prov

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



# region prov entry point
def enter_provenance_mode(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        print("[PROVENANCE]: START")
        #print("SELF ATTRIBUTES:", dir(self))
        print(f"I know you executed {func.__name__}")
        result_dataop = func(*args,**kwargs)                                        # Just execute the function and get the result
        if type(result_dataop) == skrub._data_ops._data_ops.DataOp:                 # If that is a DataOp we can inspect what is stored inside
            print("Final product is a DataOp")
            final_dict = result_dataop._skrub_impl.__dict__                         # Inspecting
            if "method_name" in final_dict.keys():                                  
                print(f"""The applied method is {final_dict["method_name"]}""")
                corresponding_provenance_function = getattr(PROVENANCE_MODULE, "provenance_"+final_dict["method_name"], None)
                if corresponding_provenance_function is None:
                    print(f"""[PROVENANCE] Can't find a provenance_{final_dict["method_name"]} in the ProvenanceModule.""")
                else:
                    print("corresponding_provenance_func is NOT None")
                    name_of_the_operation, result_df = corresponding_provenance_function(result_dataop)
                    # My Attempt to store intermediate results to avoid going into the nested structures
                    PROVENANCE_MODULE.store_provenance_of_the_dataop(result_dataop, name_of_the_operation, result_df)
                    print(f"[PROVENANCE in detail] Inspecting hashdict after {name_of_the_operation}")
                    for k,w in PROVENANCE_MODULE.hashdict.items():
                        print(f"{k}: \n The name of the df is: \n {w[0]} \n The df is: {w[1]}")
                    print("If that was unreadable an excel map is created.")
                    export_hashdict_to_excel(PROVENANCE_MODULE.hashdict)
                print("[PROVENANCE]: END")
            #print(final_dict)
        return result_dataop
    return wrapper

def set_provenance(namespace, name_of_the_function, provenance_func=enter_provenance_mode):
    skrub_eval_namespace = namespace
    name = name_of_the_function
    skrub_eval = getattr(skrub_eval_namespace,name,None)
    setattr(skrub_eval_namespace, name, provenance_func(skrub_eval))
    print(f"Set provenance for {name}")


# region checklist of functions
# To adapt more functions take a look at:
# https://skrub-data.org/stable/reference/index.html+-


