# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.8.8 64-bit (''klarna'': conda)'
#     name: python388jvsc74a57bd09f119b1d3a8a63730ae7ee508102d23a56b995c1f2248036bd772c0398b7d40e
# ---

# %%
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

# %%
try:
    _ = first_run
except NameError:
    first_run = True
    os.chdir(os.getcwd().rsplit("/", 1)[0])
    from _aux import ml

# %% [markdown]
# # Load Data

# %%
X_train, y_train = joblib.load("../data/train/preprocessed/train_features_labels.joblib.gz")

X_validation, y_validation = joblib.load("../data/train/preprocessed/validation_features_labels.joblib.gz")

# %% [markdown]
# # Hold your SMOTE for a moment
#
# SMOTE has become a ubiquitous way to handle imbalanced classes by oversampling the minority class. However, the fact that many of our features have low variance due to a lot of zero values, generating artificial samples from them can actually become quite counterproductive. Thus, we will experiment with both SMOTE and a custom undersampler that tries to capture most of the variance of the majority class. Whichever strategy yields better results for our baseline model will be the one we move forward with.

# %% [markdown]
# ### 1. Custom undersampler

# %%
X_train_maj, y_train_maj, sample_variance, sample_variance_zscore, is_significant = ml.BinaryUndersampler(n_iterations=1_000).fit(X_train, y_train)
print(sample_variance, sample_variance_zscore, is_significant)

# %%
# Create new training set
X_train_undersample = np.concatenate((X_train_maj, X_train[y_train==1]))
y_train_undersample = np.concatenate((y_train_maj, y_train[y_train==1]))

# Save the new trainig set
joblib.dump([X_train_undersample, y_train_undersample], "../data/train/preprocessed/undersampled_train_features_labels.joblib.gz")

# Check class balance
pd.Series(y_train_undersample).value_counts()

# %% [markdown]
# As we can see, classes are almost equally matched. Hopefully, our stategy will improve the baseline performance, as the new sample catches an extremely high amount of variance if compared to bootstrap results. Let the drums roll...

# %%
baseline = RandomForestRegressor().fit(X_train_undersample, y_train_undersample)

# %%
threshold_perf = pd.DataFrame(
    [
        (threshold, *confusion_matrix(y_validation, (baseline.predict(X_validation) > threshold).astype(int)).ravel())
        for threshold in np.arange(.05, 1, .05)
    ],
    columns=["threshold", "tn", "fp", "fn", "tp"]
).assign(
    precision=lambda df: df["tp"] / (df["tp"] + df["fp"]),
    recall=lambda df: df["tp"] / (df["tp"] + df["fn"]),
    f1=lambda df: 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])
)

threshold_perf.to_csv("../ml_artifacts/baseline2_model_performance.csv", index=False)

threshold_perf.query("threshold > .5")

# %% [markdown]
# Honestly, these results are better than what we had expected. As we move the threshold, we can the the "precision-recall" trade-off take place. However, note that the F1 score does continually improve, which is a sympton of the fact that the trade-off is not perfectly squred in this case -- as it rarely is.
#
# Make no mistake, these are not good prediction results by any stretch of the imagination. Nevertheless, they do suggest that our strategy is successfull as baseline performance improved significantly with no change to the model, only the data changed. Let's compare them to the previous baseline.

# %%
pd.read_csv("../ml_artifacts/baseline_model_performance.csv").query("threshold > .5")

# %% [markdown]
# Contrary to what observed earlier, the F1 score drops as we move up the threshold, which is a symptom of the behaviour induced by the model. The model flags very little, which is good in the perpective of customer experience but at the cost of losing too much money for the company. In fact, such model doesn't even justify the cost of developing and maintaining it.
#
# As we get good results from our undersampling strategy and due to time constraints, we choose not to explore how SMOTE would perform at this time. Instead, we decide to allocate more time for hyperparameter tuning and model selection next.