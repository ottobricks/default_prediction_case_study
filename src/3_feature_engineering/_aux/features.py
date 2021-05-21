import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _get_worst_status_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[
            [
                "account_worst_status_0_3m",
                "account_worst_status_3_6m",
                "account_worst_status_6_12m",
            ]
        ]
        .max(axis=1)
        .combine_first(df.assign(dummy=1).dummy)
        .values.reshape(-1, 1)
    )


def _is_merchant_category_blacklisted(df: pd.DataFrame) -> np.ndarray:
    return (
        df["merchant_category"].isin(
            [
                "Tobacco",
                "Sex toys",
                "Plants & Flowers",
                "Dating services",
            ]
        )
    ).astype(int)


class ExtraColumnCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        return df.assign(
            account_worst_status_0_12m=_get_worst_status_agg(df),
            num_arch_dc_0_12m_binned=pd.cut(
                df["num_arch_dc_0_12m"], [-1, 1, 5, np.inf], labels=False
            ),
            is_merchant_category_blacklisted=_is_merchant_category_blacklisted(df),
            is_last_arch_worst_status_possible=(
                df["status_last_archived_0_24m"]
                == df["status_last_archived_0_24m"].max()
            ).astype(int),
            is_account_worst_status_0_12m_normal=lambda frame: (
                frame["account_worst_status_0_12m"] == 1
            ).astype(int),
            num_active_div_by_paid_inv_0_12m_is_above_1=(
                df["num_active_div_by_paid_inv_0_12m"] > 1
            ).astype(int),
        )
