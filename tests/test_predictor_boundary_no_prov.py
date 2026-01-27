# tests/test_predictor_boundary_no_prov.py
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin

from rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance


class SpyRegressor(BaseEstimator, RegressorMixin):
    """
    Minimal sklearn-compatible regressor that records the columns it receives.
    """
    def __init__(self):
        self.seen_columns_ = None

    def fit(self, X, y):
        # X is typically a pandas DataFrame in skrub pipelines
        self.seen_columns_ = list(getattr(X, "columns", []))
        return self

    def predict(self, X):
        # Not used by skrub preview evaluation, but keep for API completeness
        import numpy as np
        return np.zeros(len(X), dtype=float)


def test_T14_predictor_boundary_strips_provenance(restore_skrub_monkeypatch):
    enable_why_data_provenance()

    df = pd.DataFrame(
        {
            "country": ["US", "IT", "FR", "IT"],
            "age": [25, 40, 50, 35],
        }
    )
    y_df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})

    X = skrub.var("X_table", df)
    y = skrub.var("y_table", y_df)["y"]

    # Force provenance columns to exist (Var.compute gets patched)
    # and create a typical ML matrix
    X_vec = X.skb.apply(skrub.TableVectorizer())

    spy = SpyRegressor()
    model_op = X_vec.skb.apply(spy, y=y)

    # Trigger evaluation (this should call fit)
    _ = model_op.skb.preview()

    # IMPORTANT: depending on skrub behavior, the estimator may be cloned/wrapped.
    # The instance used during evaluation might not be `spy`. Try to fetch it from the DataOp.
    impl = model_op._skrub_impl
    est_used = getattr(impl, "estimator_", None)
    if est_used is None:
        # fallback: maybe stored as estimator
        est_used = getattr(impl, "estimator", None)

    # Prefer the estimator actually used by skrub
    seen = getattr(est_used, "seen_columns_", None) or getattr(spy, "seen_columns_", None)

    assert seen is not None, "Regressor did not record seen columns (fit may not have run)."
    assert not any(str(c).startswith("_prov") for c in seen), (
        f"Predictor received provenance columns: {seen}"
    )
