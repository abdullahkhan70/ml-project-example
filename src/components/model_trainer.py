import os, sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, get_params
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    train_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, true, predict):
        r2_scores = r2_score(true, predict)
        return r2_scores

    def initiate_model_trainer(self, train_arr: list, test_arr: list, preprocessor_path):
        try:
            logging.info(f"Spliting training and test datasets!")
            # X_train, y_train, X_test, y_test = train_test_split(train_arr, test_arr, test_size=0.3, random_state=42)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "random_forest": RandomForestRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "linear_regression": LinearRegression(),
                "k_neighbour": KNeighborsRegressor(),
                "xgb_regressor": XGBRegressor(),
                "catboosting_regressor": CatBoostRegressor(),
                "adaboost_regressor": AdaBoostRegressor()
            }

            params = get_params()

            model_list:dict = {}
            r2_score_list = []

            for i in range(len(list(models))):
                model = list(models.values())[i]
                param = params[list(models.keys())[i]]

                # GridSearchCV
                grid_search = GridSearchCV(model, param, cv=3)
                grid_search.fit(X_train, y_train)
                
                model.set_params(**grid_search.best_params_)
                model.fit(X_train, y_train)

                # Make Predictations.
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Metrics
                r2_score_y_train = self.evaluate_model(y_train, y_train_pred)
                r2_score_y_test = self.evaluate_model(y_test, y_test_pred)

                model_list[list(models.keys())[i]] = r2_score_y_test
                r2_score_list.append(r2_score_y_test)
            
            # To get best model score from list
            print(f"Best Model Score: {model_list}")
            best_model_score = max(sorted(model_list.values()))
            print(f"Best Model Score: {best_model_score}")

            # To get best model name
            best_model_name = list(model_list.keys())[
                list(model_list.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found yet. Please, change the hyperparameters values of the models.", sys)

            logging.info(f"Best Model found: {best_model}")

            """
            We can load our preprocessing pickle file for getting better results w.r.t to upcoming any new data, 
            but we won't be do it.
            """

            save_object(file_path=self.model_trainer_config.train_model_path, objects=best_model)

            print(f"Predicted of Best Model: {best_model.predict(X_test)}")

            print(f"R2 Score of the Best Model: {r2_score(y_test, best_model.predict(X_test))}")

            # pd.DataFrame(list(zip(model_list, r2_score_list)), columns=["model_name", "r2_scores"]).sort_values(by="r2_scores", ascending=False)
            
            return r2_score(y_test, best_model.predict(X_test))
        except Exception as error:
            raise CustomException(error, sys)
