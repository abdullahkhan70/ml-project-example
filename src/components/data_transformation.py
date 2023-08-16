import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
# from src.components.data_ingestion import DataIngestion
from src.utils import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

@dataclass
class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self):
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot_encoder", OneHotEncoder())])
            logging.info("Numerical and Categorical for Standard Scalar and One Hot Encoder is completed")
            logging.info(f"Numerical Columns: {numerical_pipeline}")
            logging.info(f"Categorical Columns: {categorical_pipeline}")
            preprocessor = ColumnTransformer([
                ("numerical_pipeline", numerical_pipeline, numerical_features),
                ("categorical_pipeline", categorical_pipeline, categorical_features)
            ])
            return (preprocessor, numerical_features)
        except Exception as error:
            raise CustomException(error, sys)

    def initiate_data_transform(self, train_path, test_path):
        try:
            if len(train_path) > 0 and len(test_path) > 0:
                train_data = pd.read_csv(train_path)
                test_data = pd.read_csv(test_path)
                logging.info(f"Read the train and test dataset completed!")
                logging.info(f"Obtaining Preprocessing Object.")

                preprocessor_object, numerical_data = self.get_transformer_obj()

                target_column = "math_score"

                input_feature_train_data = train_data.drop(columns=[target_column], axis=1)
                target_feature_train_data = train_data[target_column]

                input_feature_test_data = test_data.drop(columns=[target_column])
                target_feature_test_data = test_data[target_column]

                logging.info(f"Applying Preprocessing object in Training DataFrame and Testing DataFrame.")
                
                input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_data)
                # input_feature_test_arr = preprocessor_object.transform(input_feature_test_data)
                input_feature_test_arr = preprocessor_object.transform(input_feature_test_data)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_data)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_data)
                ]

                logging.info(f"Successfully, saved the Preprocessing object.")

                save_object(
                    file_path = self.data_transformation_config.preprocessor_object_file_path,
                    objects = preprocessor_object
                )

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_object_file_path
                )

        except Exception as e:
            raise CustomException(e, sys)
    

# if __name__ == "__main__":
#     data_transformer = DataTransformation()
#     data_ingestion = DataIngestion()
#     train_path, test_path = data_ingestion.initiate_data_ingestion()
#     data_transformer.initiate_data_transform(train_path, test_path)
#     logging.info(f"Preprocessor: {data_transformer}")