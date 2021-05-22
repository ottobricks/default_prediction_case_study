from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


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
        cv=5,
        n_jobs=None,
        verbose=1,
        scoring=None,
        refit=False,
    ) -> None:
        """

        Parameters
        ----------
        X : array
            The input samples.

        y : array
            The target values (class labels in classification).

        cv : int , default=5
            Determines the cross-validation splitting strategy.

        n_jobs : int , default=None
            The number of CPUs to use to do the computation.

        verbose : int , default=1
            Controls the verbosity when fitting and predicting.

        scoring : None
            Apply estimator’s score method for evaluating.

        refit : bool , default=False
            Refit an estimator using the best found parameters on the whole dataset.

        """
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

            print(k)
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
