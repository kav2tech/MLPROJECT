import pandas as pd
import numpy as np
import sys
import os
import logging
import traceback
from src.exception import CustomException
from src.utils import load_object

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class CustomPredictPipeline:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        logging.debug("Initializing CustomPredictPipeline with input values")
        logging.debug(f"gender={gender}, race_ethnicity={race_ethnicity}, "
                      f"parental_level_of_education={parental_level_of_education}, "
                      f"lunch={lunch}, test_preparation_course={test_preparation_course}, "
                      f"reading_score={reading_score}, writing_score={writing_score}")

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            logging.debug("Converting input values to DataFrame for prediction")
            input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [float(self.reading_score)],
                "writing_score": [float(self.writing_score)]
            }
            df = pd.DataFrame(input_dict)
            logging.debug(f"Generated DataFrame:\n{df}")
            return df
        except Exception as e:
            logging.error("Error in get_data_as_dataframe")
            logging.error(traceback.format_exc())
            raise CustomException(e, sys)


class PredictPipeline:
    def predict(self, features):
        try:
            logging.debug("Starting prediction process")
            artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")
            model_path = os.path.join(artifact_dir, 'model.pkl')
            preprocessor_path = os.path.join(artifact_dir, 'preprocessor.pkl')

            logging.debug(f"Model path: {model_path}")
            logging.debug(f"Preprocessor path: {preprocessor_path}")

            # Load model
            model = load_object(model_path)
            logging.debug(f"Loaded model: {type(model).__name__}")

            # Load preprocessor
            preprocessor = load_object(preprocessor_path)
            logging.debug("Preprocessor loaded successfully")

            # Transform data
            logging.debug("Transforming input features")
            data_scaled = preprocessor.transform(features)
            logging.debug(f"Scaled data shape: {data_scaled.shape}")

            # Predict
            logging.debug("Generating predictions")
            prediction = model.predict(data_scaled)

            logging.info(f"Prediction result: {prediction[0]}")
            return prediction[0], type(model).__name__

        except Exception as e:
            logging.error("Error in PredictPipeline.predict")
            logging.error(traceback.format_exc())
            raise CustomException(e, sys)

