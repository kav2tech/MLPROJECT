import os
import sys
import logging
import pandas as pd
from src.exception import CustomException
from src.custom_logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        
        try:
            df = pd.read_csv('Notebook//Data/stud.csv')
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


            logging.info("Train and test data saved successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
                raise CustomException(e, sys) from e
        
if __name__ == "__main__":  
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))
    logging.info("Data Ingestion completed successfully")

'''
#This script handles the data ingestion process for your machine learning pipeline:

Imports: Loads necessary libraries for file handling, logging, data manipulation, and splitting.

Config Class: DataIngestionConfig defines file paths for raw, train, and test data.

DataIngestion Class:

Reads the student data CSV into a DataFrame.

Saves the raw data to the artifacts folder.

Splits the data into training (80%) and testing (20%) sets.

Saves these splits as separate CSV files in the artifacts folder.

Logs each step for traceability.

Returns the paths to the train and test data files.

Error Handling: If any error occurs, it raises a custom exception for better debugging.

Main Block: Runs the ingestion process when the script is executed directly.

Summary:
This code automates reading, splitting, and saving your dataset, preparing it for further steps in your ML workflow.
'''