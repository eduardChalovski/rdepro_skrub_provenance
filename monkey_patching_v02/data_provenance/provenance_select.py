import skrub
import pandas as pd
from skrub import SelectCols

def main():
    from monkey_patching_v02_data_provenance import set_provenance, enter_provenance_mode_dataop, enter_provenance_mode_var
    set_provenance(skrub._data_ops._evaluation,"evaluate", provenance_func=enter_provenance_mode_dataop)
    set_provenance(skrub._data_ops._data_ops.Var,"compute", provenance_func=enter_provenance_mode_var)


    df = pd.DataFrame({"A": [1, 2], "B": [10, 20], "C": ["x", "y"], "_prov1": [1,2]})
    df_var = skrub.var("df_var", df)
    print(df_var.skb.apply(SelectCols(["C", "A"])).skb.eval())
    
# python .\monkey_patching_v02\data_provenance\provenance_select.py

if __name__ == "__main__":
    main()