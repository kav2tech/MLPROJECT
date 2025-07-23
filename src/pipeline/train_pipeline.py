import os
import sys
from src.Exception import CustomException
from src.Custom_logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    try:
        logging.info("▶ Starting the training pipeline...")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        logging.info(f"Data ingestion complete.\nTrain path: {train_path}\nTest path: {test_path}")

        # 2. Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

        logging.info(" Data transformation complete. Transformed arrays ready.")

        # 3. Model Training
        trainer = ModelTrainer()
        best_model, model_report = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("Model training complete.")
        logging.info(f" Best model details:\n{model_report}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()
