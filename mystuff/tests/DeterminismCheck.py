# check_determinism.py
import subprocess
import sys
import os
import pickle
import pytest
from deepdiff import DeepDiff
# The script you want to test
SCRIPT = "C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/mystuff/pipeline2_SKRUBIFIED.py"
output_file = "C:/Users/teodo/Desktop/github/rdepro_skrub_provenance/output.pkl"
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

 
def test_pipeline_is_deterministic():
    print("wry")
    out1 = run_script()
    out2 = run_script()
    print("blegh")
    df1 = out1.skb.eval()  # or .preview() for a small sample
    df2 = out2.skb.eval()
    assert df1.equals(df2), "Pipeline output is non-deterministic"