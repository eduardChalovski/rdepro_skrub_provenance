from __future__ import annotations
import pandas as pd

import rdepro_skrub_provenance.constants as c


def load_simple_dataset_pandas() -> pd.DataFrame:
    return pd.read_csv(c.SIMPLE_DATASET_PATH)
