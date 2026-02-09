# check_determinism.py
import subprocess
import sys
import os
import pickle
import pandas
import numpy

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--track-provenance",
    action="store_true",
    help="Enable provenance tracking"
)
args = parser.parse_args()

if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance, evaluate_provenance
    enable_why_data_provenance()
else:
    print("Provenance is disabled")


print("Libraries imported")


def replace_nan_with_minus_one(x):
    if isinstance(x, (list, numpy.ndarray)):
        return [replace_nan_with_minus_one(i) for i in x]
    else:
        return -1 if pandas.isna(x) else x

# The script you want to test
SCRIPT = "./pipelines/BasicDataAnalysisCase.py"
output_file = "./tests/output.pkl"
# Function to run the script and capture its output
def run_script():
    if os.path.exists(output_file):
        os.remove(output_file)

    result = subprocess.run([sys.executable, SCRIPT], capture_output=True)

    if result.returncode != 0:
        print("Script failed to run")
        print(result.stderr.decode())
        sys.exit(1)
    # Assume your script saves the final DataFrame to a pickle
    with open(output_file, "rb") as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    print("Running first execution...")
    out1 = run_script()
    print("Running second execution...")
    out2 = run_script()
    # Compare outputs
    # Evaluate the DataOps to get pandas DataFrames
    df1 = out1.skb.preview().fillna(-1)  # or .preview() for a small sample
    prov_cols = [col for col in df1.columns if col.startswith("_prov")]

    # since NaN != NaN -> we need to filter them out
    # but there is also [NaN] != [NaN] -> we need an additional mapping
    for prov_col in prov_cols:
        df1[prov_col] = df1[prov_col].apply(replace_nan_with_minus_one)

    df2 = out2.skb.preview().fillna(-1)

    for prov_col in prov_cols:
        df2[prov_col] = df2[prov_col].apply(replace_nan_with_minus_one)


# Compare the actual data
    if df1.equals(df2):
        print("✅ Deterministic: outputs are identical")
    else:
        print("❌ Non-deterministic: outputs differ")