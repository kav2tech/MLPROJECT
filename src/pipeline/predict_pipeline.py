import pandas as pd
import numpy as np
import sys
import os

from src.Exception import CustomException
from src.utils import load_object

class CustomPredictPipeline:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
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
            print("[INFO] Custom input DataFrame created:")
            print(df)
            return df

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            print("[INFO] Scaled data:")
            print(data_scaled)
            print(f"[INFO] Prediction: {prediction[0]}")

            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)
             