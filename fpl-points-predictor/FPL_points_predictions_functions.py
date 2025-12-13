from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_poisson_deviance, d2_tweedie_score
import pandas as pd
import numpy as np

class ModelSelector:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def train_test(self, target="total_points"):
        X = self.data.drop(columns=["date", "element", "opponent_team", "season", target],
                           errors='ignore').select_dtypes(exclude="object")

        train_mask = ~self.data['season'].isin(['2425', '2526'])
        test_mask = self.data['season'] == '2425'

        train_min = self.data.loc[train_mask, target].min()
        test_min = self.data.loc[test_mask, target].min()

        SHIFT_VALUE = max(np.abs(train_min), np.abs(test_min))

        return (X[train_mask],
                X[test_mask],
                self.data.loc[train_mask, target] + SHIFT_VALUE,
                self.data.loc[test_mask, target] + SHIFT_VALUE)

    @staticmethod
    def metrics_info(self, y_pred: np.ndarray, y_test: np.ndarray):
        mpd = mean_poisson_deviance(y_true=y_test,
                                    y_pred=y_pred)
        d2 = d2_tweedie_score(y_true=y_test,
                              y_pred=y_pred,
                              power=1)
        return mpd, d2

    @staticmethod
    def params_search(
                      models: list,
                      models_names: list,
                      params_grid: list,
                      X_train: np.ndarray,
                      y_train: np.ndarray):
        results_list = []
        best_models_map = {}
        tscv = TimeSeriesSplit(n_splits=5)

        for model, model_name, grid in zip(models, models_names, params_grid):
            random_search = RandomizedSearchCV(cv=tscv,
                                               n_iter=10,
                                               estimator=model,
                                               scoring="neg_mean_poisson_deviance",
                                               param_distributions=grid,
                                               verbose=1,
                                               error_score="raise",
                                               random_state=77,
                                               refit=True)
            random_search.fit(X_train, y_train)
            results_list.append({
                "model": model_name,
                "best_params": random_search.best_params_,
                "fit_time": random_search.refit_time_})
            best_models_map[model_name] = random_search.best_estimator_

        return pd.DataFrame(results_list), best_models_map

    def models_validation(self, trained_models_map, X_test, y_test):
        results_list = []
        for model_name, model in trained_models_map.items():
            y_pred = model.predict(X_test)

            metrics = self.metrics_info(y_pred=y_pred, y_test=y_test)

            row = {
                "model_name": model_name,
                "mpd": metrics[0],
                "d2": metrics[1]}
            results_list.append(row)

        return pd.DataFrame(results_list)