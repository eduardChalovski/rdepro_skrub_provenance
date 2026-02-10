from __future__ import annotations
import pandas as pd
import numpy as np

# Helper Integers shifted
class TableRegistry:
    def __init__(self, start: int = 0):
        self._name_to_id = {}
        self._id_to_name = {}
        self._next_id = start

    def get_id(self, table_name: str) -> int:
        if table_name not in self._name_to_id:
            tid = self._next_id
            self._name_to_id[table_name] = tid
            self._id_to_name[tid] = table_name
            self._next_id += 1
        return self._name_to_id[table_name]

    def get_name(self, table_id: int) -> str:
        return self._id_to_name[table_id]

TABLE_REGISTRY = TableRegistry()

# Integers shifted: 16 most left bits are reserved for tables -> support of 65 536 tables
# -> 48 bits for rows per table -> 281 trillion rows per table
def with_provenance_integers_shifted(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    df = df.copy()
    table_id = TABLE_REGISTRY.get_id(table_name)
    row_ids = np.arange(len(df), dtype=np.int64)            # TODO: consider using indices instead of len(df)
    df["_prov" + str(table_id)] = (np.int64(table_id) << 48) | row_ids      # pack_prov(table_id, row_id)
    return df


def pack_prov(table_id: int, row_id: np.ndarray) -> np.ndarray:
    return (np.int64(table_id) << 48) | row_id


# support of our fancy provenance:
def decode_prov(prov):
    table_id = prov >> 48
    row_id = prov & ((1 << 48) - 1)
    table = TABLE_REGISTRY.get_name(table_id)
    return f"{table}:{row_id}"