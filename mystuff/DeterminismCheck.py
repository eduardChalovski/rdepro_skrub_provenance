# check_determinism.py
import subprocess
import sys
import os
import pickle
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

if __name__ == "__main__":
    print("Running first execution...")
    out1 = run_script()
    print("Running second execution...")
    out2 = run_script()
    # Compare outputs
# Evaluate the DataOps to get pandas DataFrames
    df1 = out1.skb.eval()  # or .preview() for a small sample
    df2 = out2.skb.eval()

# Compare the actual data
    if df1.equals(df2):
        print("✅ Deterministic: outputs are identical")
    else:
        print("❌ Non-deterministic: outputs differ")