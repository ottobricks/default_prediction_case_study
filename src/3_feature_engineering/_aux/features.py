import numpy as np
import pandas as pd

def get_worst_status_agg(df: pd.DataFrame) -> pd.DataFrame:
    return df[
            [
                "account_worst_status_0_3m",
                "account_worst_status_3_6m",
                "account_worst_status_6_12m",
            ]
        ].max(axis=1).combine_first(df.assign(dummy=1).dummy).values.reshape(-1, 1)

def is_last_arch_worst_status_possible(df: pd.DataFrame) -> np.ndarray:
    return (df["status_last_archived_0_24m"] == df["status_last_archived_0_24m"].max()).astype(int).values.reshape(-1, 1)

def is_account_worst_status_0_12m_normal(df: pd.DataFrame) -> np.ndarray:
    return (
        get_worst_status_agg(df) == 1
    ).astype(int).reshape(-1, 1)

def num_active_div_by_paid_inv_0_12m_is_above_1(df: pd.DataFrame) -> np.ndarray:
    return (df["num_active_div_by_paid_inv_0_12m"] > 1).astype(int).values.reshape(-1, 1)