from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_poisson_deviance, d2_tweedie_score, mean_absolute_error
import pandas as pd
import numpy as np

class ModelSelector:
    def __init__(self, data: pd.DataFrame):

        self.data = data

    def train_test(self, target="total_points"):
        """
        :param target:
        train contains  seasons: ['2021', '2122', '2223', '2322']
        test contains  seasons: ['2425', '2526']
        :return: X_train, X_test, y_train, y_test
        """
        X = self.data.drop(columns=["date", "element", "opponent_team", "season", target],
                           errors='ignore').select_dtypes(exclude="object").copy()

        mask = self.data['season'].isin(['2425', '2526'])
        train_mask = ~mask
        test_mask = mask

        train_min = self.data.loc[train_mask, target].min()
        test_min = self.data.loc[test_mask, target].min()

        SHIFT_VALUE =  max(np.abs(train_min), np.abs(test_min))
        return (X[train_mask],
                X[test_mask],
                self.data.loc[train_mask, target] + SHIFT_VALUE,
                self.data.loc[test_mask, target] + SHIFT_VALUE)

    @staticmethod
    def metrics_raport(y_pred: np.ndarray, y_true: np.ndarray):
        """
        :param y_pred:
        :param y_true:
        :return: mean_poisson_deviance, d2_tweedie_score, mean_absolute_error
        """
        mpd = mean_poisson_deviance(y_true=y_true,
                                    y_pred=y_pred)
        d2 = d2_tweedie_score(y_true=y_true,
                              y_pred=y_pred,
                              power=1)
        mae = mean_absolute_error(y_true=y_true,
                                  y_pred=y_pred)
        return mpd, d2, mae

    def params_search(self,
                    models: list,
                    models_names: list,
                    params_grid: list,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    penalty_value: float = 2,
                    n_iter: int = 10):
        """
        :param models:
        :param models_names:
        :param params_grid:
        :param X_train:
        :param y_train:
        :param penalty_value:
        :param n_iter:

        :return: results_list, best_models_map
        results_list: best models' metrics on train set
        best_models_map: best models
        """
        results_list = []
        best_models_map = {}
        tscv = TimeSeriesSplit(n_splits=5)


        sample_weights = np.ones(len(X_train))
        sample_weights[np.array(y_train)==0] = penalty_value

        for model, model_name, grid in zip(models, models_names, params_grid):
            random_search = RandomizedSearchCV(cv=tscv,
                                               n_iter=n_iter,
                                               estimator=model,
                                               scoring="neg_mean_poisson_deviance",
                                               param_distributions=grid,
                                               verbose=1,
                                               error_score="raise",
                                               random_state=77,
                                               refit=True)
            try:
                random_search.fit(X_train, y_train, sample_weight=sample_weights)

            except (ValueError, TypeError):
                random_search.fit(X_train, y_train)
            best_models_map[model_name] = random_search.best_estimator_
            metrics = self.metrics_raport(y_true=y_train,
                                        y_pred=random_search.best_estimator_.predict(X_train))
            row = {
                "data": "train",
                "model_name": model_name,
                "best_params": random_search.best_params_,
                "mpd": metrics[0],
                "d2": metrics[1],
                "mae": metrics[2]}
            results_list.append(row)
        return pd.DataFrame(results_list), best_models_map
    def models_validation(self, trained_models_map, X_test, y_test):
        """
        :param trained_models_map:
        :param X_test:
        :param y_test:
        :return: results_list
        results_list: best models' metrics on test set
        """
        results_list = []
        for model_name, model in trained_models_map.items():
            y_pred = model.predict(X_test)

            metrics = self.metrics_raport(y_pred=y_pred, y_true=y_test)

            row = {
                "data" : "test",
                "model_name": model_name,
                "mpd": metrics[0],
                "d2": metrics[1],
                "mae": metrics[2]}
            results_list.append(row)

        return pd.DataFrame(results_list)
