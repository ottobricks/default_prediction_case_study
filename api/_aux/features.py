from typing import List
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
        .fillna(1)
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


def required_columns() -> List[str]:
    return ['uuid', 'account_amount_added_12_24m', 'account_days_in_dc_12_24m',
       'account_days_in_rem_12_24m', 'account_days_in_term_12_24m',
       'account_incoming_debt_vs_paid_0_24m', 'account_status',
       'account_worst_status_0_3m', 'account_worst_status_12_24m',
       'account_worst_status_3_6m', 'account_worst_status_6_12m', 'age',
       'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'merchant_category',
       'merchant_group', 'has_paid', 'max_paid_inv_0_12m',
       'max_paid_inv_0_24m', 'name_in_email',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
       'num_arch_ok_12_24m', 'num_arch_rem_0_12m',
       'num_arch_written_off_0_12m', 'num_arch_written_off_12_24m',
       'num_unpaid_bills', 'status_last_archived_0_24m',
       'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
       'status_max_archived_0_6_months', 'status_max_archived_0_12_months',
       'status_max_archived_0_24_months', 'recovery_debt',
       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
       'sum_paid_inv_0_12m', 'time_hours', 'worst_status_active_inv']


def expected_payload():
    return """'[{"uuid":"1229c83c-6338-4c4b-a20f-065ecca45b4a",
    "account_amount_added_12_24m":28472,
    "account_days_in_dc_12_24m":0.0,
    "account_days_in_rem_12_24m":0.0,
    "account_days_in_term_12_24m":0.0,
    "account_incoming_debt_vs_paid_0_24m":0.0,
    "account_status":1.0,
    "account_worst_status_0_3m":1.0,
    "account_worst_status_12_24m":1.0,
    "account_worst_status_3_6m":1.0,
    "account_worst_status_6_12m":1.0,
    "age":29,
    "avg_payment_span_0_12m":8.24,
    "avg_payment_span_0_3m":7.8333333333,
    "merchant_category":"Diversified electronics",
    "merchant_group":"Electronics",
    "has_paid":true,
    "max_paid_inv_0_12m":37770.0,
    "max_paid_inv_0_24m":37770.0,
    "name_in_email":"F1+L",
    "num_active_div_by_paid_inv_0_12m":0.037037037,
    "num_active_inv":1,
    "num_arch_dc_0_12m":0,
    "num_arch_dc_12_24m":0,
    "num_arch_ok_0_12m":25,
    "num_arch_ok_12_24m":16,
    "num_arch_rem_0_12m":0,
    "num_arch_written_off_0_12m":0.0,
    "num_arch_written_off_12_24m":0.0,
    "num_unpaid_bills":1,
    "status_last_archived_0_24m":1,
    "status_2nd_last_archived_0_24m":1,
    "status_3rd_last_archived_0_24m":1,
    "status_max_archived_0_6_months":1,
    "status_max_archived_0_12_months":1,
    "status_max_archived_0_24_months":1,
    "recovery_debt":0,
    "sum_capital_paid_account_0_12m":116,
    "sum_capital_paid_account_12_24m":27874,
    "sum_paid_inv_0_12m":265347,
    "time_hours":14.1708333333,
    "worst_status_active_inv":1.0}]'"""

class ExtraColumnCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        return df.assign(
            account_worst_status_0_12m=_get_worst_status_agg(df),
            num_arch_dc_0_12m_binned=pd.cut(
                df["num_arch_dc_0_12m"].fillna(0), [-1, 1, 5, np.inf], labels=False
            ),
            is_merchant_category_blacklisted=_is_merchant_category_blacklisted(df),
            is_last_arch_worst_status_possible=(
                df["status_last_archived_0_24m"].fillna(1)
                == df["status_last_archived_0_24m"].fillna(1).max()
            ).astype(int),
            is_account_worst_status_0_12m_normal=lambda frame: (
                frame["account_worst_status_0_12m"].fillna(1) == 1
            ).astype(int),
            num_active_div_by_paid_inv_0_12m_is_above_1=(
                df["num_active_div_by_paid_inv_0_12m"].fillna(0) > 1
            ).astype(int),
        )
