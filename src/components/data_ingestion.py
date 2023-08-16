import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"Entered the adata ingesiton method or component")
        try:
            data = pd.read_csv('research\data\stud.csv')
            logging.info(f"Read the dataset as DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Train test Split initialized")
            training_dataset, testing_dataset = train_test_split(data, test_size=0.2, random_state=42)
            training_dataset.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            testing_dataset.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed!")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessing_path = data_transformation.initiate_data_transform(train_path, test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessing_path)
