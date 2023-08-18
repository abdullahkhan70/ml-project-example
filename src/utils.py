import os, sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.compose import ColumnTransformer

def save_object(file_path: str, objects):
    try:
        dir_path = os.path.dirname(file_path)
        print(f"File Path: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(objects, file_obj)

    except Exception as error:
        raise CustomException(error, sys)

def load_object(model_path: str):
    try:
        dir_path = os.path.dirname(model_path)
        print(f"Model Path: {dir_path}")

        with open(model_path, "rb") as file_obj:
            data = dill.load(file=file_obj)
        return data
        
    except Exception as error:
        raise CustomException(error, sys)

def get_params():
    params = {
        "random_forest": {
            'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            'max_depth': [None, 2, 4, 6, 8],
            "max_features": [None, "sqrt", "log2"],
            "random_state": [42],
            "verbose": [False],
            # "max_sample": [4, 8, 12, 16]
        },
        "decision_tree": {
            'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "max_depth": [2, 5, 10],
            'splitter': ["best", "random"],
            'random_state': [42],
            'max_features': [4, 8, 12, 16, 20]
        },
        'gradient_boosting': {
            'loss': ["squared_error", "absolute_error", "huber", "quantile"],
            # 'learning_rate': [0.01, 0.078, 0.2, 0.1],
            'n_estimators': [8, 16, 33, 64, 128],
            # 'subsample': [0.25, 0.36, 1.0],
            # 'criterion': ["friedman_mse", "squared_error"],
            # 'max_depth': [2, 3, 5, 8],
            'random_state': [42], 
            'max_features': ["sqrt", 'log2', None],
            'verbose': [False]
        }, 
        'linear_regression': {
            'fit_intercept': [True],
            'copy_X': [False],
            'n_jobs': [None, 2, 5]
        },
        'k_neighbour': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30, 50, 65],
            'n_jobs': [None, 2, 5, 7]
        },
        'xgb_regressor': {
            'booster': ['gbtree', 'gblinear'],
            'validate_parameters': [True],
            # 'eta': [0.2, 0.01, 0.025],
            'max_depth': [6, 8, 12]
        }, 
        'adaboost_regressor': {
            'n_estimators': [45, 50, 80, 120],
            # 'learning_rate': [0.25, 0.68, 1.0],
            'loss': ["linear", 'square', 'exponential'],
            'random_state': [42]
        },
        "catboosting_regressor": {
            'verbose': [False]
        }
    }

    return params