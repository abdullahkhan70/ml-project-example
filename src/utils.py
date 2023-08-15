import os, sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.compose import ColumnTransformer

def save_object(file_path: str, object):
    try:
        dir_path = os.path.dirname(file_path)
        print(f"File Path: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as error:
        raise CustomException(error, sys)