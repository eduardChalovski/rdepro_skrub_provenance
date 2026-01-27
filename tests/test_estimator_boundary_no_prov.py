# tests/test_estimator_boundary_no_prov.py
import pandas as pd
import skrub

from sklearn.base import BaseEstimator, TransformerMixin

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import (
    enable_why_data_provenance,
)

# Global capture store (works even if the estimator is cloned/wrapped)
CAPTURED_COLUMNS = []


class SpyTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that records the columns it sees.

    Important: skrub/sklearn may clone/wrap the estimator, so we cannot rely
    on instance attributes. We record into a global list instead.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            CAPTURED_COLUMNS.append(list(X.columns))
        else:
            CAPTURED_COLUMNS.append(None)
        return X


def test_T07_transformer_boundary_strips_provenance(restore_skrub_monkeypatch):
    enable_why_data_provenance()
    CAPTURED_COLUMNS.clear()

    df = pd.DataFrame(
        {
            "country": ["US", "IT", "FR", "IT"],
            "age": [25, 40, 50, 35],
        }
    )

    X = skrub.var("X_table", df)

    # Keep DataFrame semantics (no TableVectorizer here)
    X2 = X.assign(dummy=1)

    X_out = X2.skb.apply(SpyTransformer())

    # Trigger execution
    _ = X_out.skb.preview()

    assert CAPTURED_COLUMNS, "SpyTransformer.transform() was not executed."

    seen = CAPTURED_COLUMNS[-1]
    assert seen is not None, "SpyTransformer did not receive a DataFrame."

    assert not any(col.startswith("_prov") for col in seen), (
        "Provenance columns (_prov*) were passed to a sklearn transformer.\n"
        f"Columns seen: {seen}"
    )
