import pandas as pd

BASE = "benchmark_logs/"
# Input files
NO_PROV_FILE = f"{BASE}benchmark_runtime_n_rows_no_provenance.csv"
LIST_FILE = f"{BASE}benchmark_runtime_n_rows_provenance_list.csv"

# Output file
OUT_FILE = f"{BASE}benchmark_runtime_absolute_overhead_list.csv"

# Columns for which we compute overhead
COLUMNS = [
    "tablevectorizer_time",
    "standardScaler_time",
    "oneHotEncoder_time",
]

# Load CSVs
df_no = pd.read_csv(NO_PROV_FILE)
df_list = pd.read_csv(LIST_FILE)

# Merge on number of input rows
df = df_list.merge(
    df_no,
    on="n_rows_input",
    suffixes=("_list", "_no")
)

# Compute absolute overheads
out = pd.DataFrame()
out["n_rows_input"] = df["n_rows_input"]

for col in COLUMNS:
    out[f"{col}_overhead"] = df[f"{col}_list"] - df[f"{col}_no"]

# Save result
out.to_csv(OUT_FILE, index=False)

print(f"Wrote absolute overhead CSV to: {OUT_FILE}")
