import math
import multiprocessing as mp
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ZSCORE_CRITICAL_VALUE = 1.645 # equivalent to p-value = 0.05

class EstimatorSelector:
    """

    Parameters
    ----------
    models : dict
        Model from sklearn or xgboost to apply. The model has to be
        associated with a sklearn or xgboost object in the form ``{'model_name': model()}``.

    params : dict
        Parameters from the models selected in the form ``{'model_name': {'param1': [], 'param2': [], ...}}``.

    timeseries_data : bool, default=False
        Specify whether or not the data is a time series.

    Attributes
    ----------
    keys : TODO

    grid_searches : TODO

     Notes
    -----

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xgboost.sklearn import XGBClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> models = {"RandomForestClassifier": RandomForestClassifier(),
    ...           "XGBoost": XGBClassifier(objective="binary:logistic")}
    >>> params = {"RandomForestClassifier": {"n_estimators": [100, 200],
    ...                                      "criterion": ["gini", "entropy"],
    ...                                      "max_features": [5],
    ...                                      "max_depth": [20],
    ...                                      "min_samples_split": [4],
    ...                                      "bootstrap": [True]},
    ...           "XGBoost": {"min_child_weight": [1, 5],
    ...                       "gamma": [0.5, 1, 1.5],
    ...                       "subsample": [0.6, 0.8],
    ...                       "colsample_bytree": [0.8, 1.0],
    ...                       "max_depth": [3, 5]}}
    >>> selector = EstimatorSelector(models, params)
    >>> selector.fit(X, y, scoring="f1", n_jobs=-1)
    >>> selector.score_summary(sort_by="mean_score")
    """

    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        params: Dict[str, Dict[str, List[Any]]],
    ) -> None:
        """ """
        if not set(models.keys()).issubset(set(params.keys())):

            missing_params = list(set(models.keys()) - set(params.keys()))

            raise ValueError(
                "Some estimators are missing parameters: {}".format(missing_params)
            )

        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int=5,
        n_jobs: int=None,
        verbose: int=1,
        scoring=None,
        refit: bool=False,
        imbalanced_data: bool=False,
        random_state: int=None,
        shuffle_folds: bool=False
    ) -> None:
        """

        Parameters
        ----------
        X : array
            The input samples.

        y : array
            The target values (class labels in classification).

        cv : int, cv_splitter, iterable, default=5
            Cross-validation splitting strategy. See [GridSearch][1] for more information

        n_jobs : int , default=None
            The number of CPU cores to use for computation.

        verbose : int , default=1
            Controls the verbosity when fitting and predicting.

        scoring : str, callable, list, tuple or dict, default=None
            Strategy to evaluate the performance of the cross-validated model on the test set.

        refit : bool , default=False
            Refit an estimator using the best found parameters on the whole dataset.


        [1]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        """
        if imbalanced_data:
            cv = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=shuffle_folds)

        for key in self.keys:
            print("Running GridSearchCV for {}.".format(key))
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(
                model,
                params,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=refit,
                return_train_score=True,
            )
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by="mean_score") -> pd.DataFrame:
        """

        Parameters
        ----------
        sort_by : string
            A data frame column in which a sort function will be applied.

        """
        rows = []

        def get_results_row(key, scores, params):
            d = {
                "estimator": key,
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
            }
            return pd.Series({**{"params": params}, **d})

        for k in self.grid_searches:

            params = self.grid_searches[k].cv_results_["params"]
            scores = []

            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)

            for p, s in zip(params, all_scores):
                rows.append((get_results_row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ["estimator", "min_score", "mean_score", "max_score", "std_score"]

        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


class BinaryUndersampler:
    
    def __init__(self, n_iterations: int=5, n_jobs: int=-1):
        self.n_iterations = n_iterations
        self.highest_variance_sample = (None, None)
        self.n_jobs = min(mp.cpu_count(), n_jobs) if n_jobs > 1 else mp.cpu_count()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """Function to find a subsample of majority class (label) and its significance

        Return
        ------
        Tuple[
            X_subsample, numpy.ndarray: The selected subsample of the majority class
            y_subsample, numpy.ndarray: The respective labels of the subsample
            sample_variance, float: The subsample mean variance (mean of variance of each feature)
            sample_variance_zscore, float: The bootstrap Z-score of the `sample_variance` from `n_iterations`,
            is_sample_var_significantly_higher, bool: Whether the `sample_variance` is significantly higher than the bootstrap mean (p-value < 0.05)
        ]
        """
        self.majority_class = pd.value_counts(pd.Series(y)).index[0]
        self.n_splits = math.floor(X.shape[0] / np.sum(y!=self.majority_class))
        # split and select the sample with highest variance (decide how significant via bootstraping)
        with mp.Pool(self.n_jobs) as pool:
            bootstrap_samples = pool.starmap(
                self.get_highest_variance_sample,
                [(X[y==self.majority_class], y[y==self.majority_class], self.n_splits, idx) for idx in range(self.n_iterations)]
            )
        highest_zscore = sorted(stats.zscore(np.array(list(map(lambda tup: tup[-1], bootstrap_samples)))))[::-1][0]
        is_sample_var_significantly_higher = highest_zscore > ZSCORE_CRITICAL_VALUE
        highest_mean_var_sample = sorted(bootstrap_samples, key=lambda tup: tup[-1])[::-1][0]
        return (*highest_mean_var_sample, highest_zscore, is_sample_var_significantly_higher)
        

    def get_highest_variance_sample(self, X: np.ndarray, y_label: np.ndarray, n_splits: int, random_state: int=42):
        disjoint_samples = self._shuffle_split_observations(X, y_label, n_splits, random_state)
        disjoint_samples_mean_var = map(self._get_mean_feature_variance, disjoint_samples)
        highest_mean_var_sample = sorted(disjoint_samples_mean_var, key=lambda tup: tup[-1])[::-1][0]
        return highest_mean_var_sample

    def _shuffle_split_observations(self, X: np.ndarray, y_label: np.ndarray, n_splits: int, random_state: int=42) -> List:
        """Return a list of disjoint samples with the `y_label` concatenated at axis=1 (column)"""
        np.random.seed(random_state)
        return np.array_split(np.random.permutation(np.concatenate((X, y_label.reshape(-1, 1)), axis=1)), n_splits)

    def _get_mean_feature_variance(self, sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        X, y_label = sample[:, :-1], sample[:, -1]
        mean_variance = np.mean(np.ma.var(X, axis=0))
        return (X, y_label, mean_variance)
