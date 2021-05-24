import pandas as pd
from scipy.stats import chi2


def test_k_prop(c_r_table: pd.DataFrame, alpha: float = 0.05, theta_null=None):
    """
    Function to compute expected values and chi2 scores

    Code adapted from https://github.com/NadimKawwa/Statistics/blob/master/Hypothesis%20Testing%20For%20Proportions.ipynb
    """

    # calculcate degrees of freedom
    deg_freedom = c_r_table.shape[0]

    # check if theta specified
    if theta_null:
        theta = theta_null

    # calculcated pooled estimate otherwise
    else:
        # reduce deg of freedom
        deg_freedom -= 1
        print("Using {} degrees of freedom".format(deg_freedom))
        theta_hat = c_r_table["default"].sum() / c_r_table["count"].sum()
        theta = theta_hat

    # calculcate cutoff
    chi_critical = chi2.isf(alpha, deg_freedom)

    # calculated expected values:
    expected_default = c_r_table["count"] * theta
    e_lose = c_r_table["count"] * (1 - theta)

    # create copy of dataframe
    df_test = c_r_table.copy()
    df_test["expected_default"] = expected_default
    df_test["expected_not_default"] = e_lose

    df_test["chi_default"] = (
        (df_test["default"] - df_test["expected_default"]) ** 2
    ) / df_test["expected_default"]

    df_test["chi_not_default"] = (
        (df_test["not_default"] - df_test["expected_not_default"]) ** 2
    ) / df_test["expected_not_default"]

    chi_test = df_test["chi_default"].sum() + df_test["chi_not_default"].sum()

    if chi_test > chi_critical:
        print("Reject null hypothesis with {} > {}".format(chi_test, chi_critical))
    else:
        print("Maintain null hypothesis with {} < {}".format(chi_test, chi_critical))

    return chi_test, df_test


def complement(s: pd.Series):
    return s.shape[0] - s.sum()
