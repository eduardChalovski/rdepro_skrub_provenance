import numpy as np
import pandas as pd


def make_dataset(n_rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Main table
    df_main = pd.DataFrame({
        "user_id": rng.integers(0, n_rows // 10, size=n_rows),
        "category": rng.choice(["A", "B", "C", "D"], size=n_rows),
        "value": rng.normal(0, 1, size=n_rows),
        "text": rng.choice(
            [f"token_{i}" for i in range(100)],  # controls sparsity
            size=n_rows
        ),
    })

    # Lookup table for merge
    df_lookup = pd.DataFrame({
        "user_id": np.arange(n_rows // 10),
        "country": rng.choice(["FR", "DE", "US", "UK"], size=n_rows // 10),
        "segment": rng.choice(["S1", "S2", "S3"], size=n_rows // 10),
    })

    return df_main, df_lookup