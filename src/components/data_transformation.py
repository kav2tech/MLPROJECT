import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.custom_logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer for preprocessing.
        Applies scaling on numerical columns and one-hot encoding + scaling on categorical columns.
        """
        try:
            # Use cleaned column names
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            logging.info(f"Pipelines created for numerical: {numerical_columns} and categorical: {categorical_columns}")

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ])

            logging.info("ColumnTransformer created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Applies transformations to train and test data and returns transformed arrays and the path of the saved preprocessor object.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded.")

            # Normalize column names to avoid errors
            train_df.columns = train_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
            test_df.columns = test_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

            logging.info(f"Normalized column names: {train_df.columns.tolist()}")

            target_column_name = "math_score"

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Separated input and target features.")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine with target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed successfully.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {str(e)}")
            raise CustomException(e, sys)



"""
data_transformation.py

This module handles all data preprocessing steps required before training machine learning models.
It is designed to be modular, robust, and easily maintainable.

Key functionalities:
- Defines a configuration dataclass for transformation-related file paths.
- Implements a DataTransformation class that:
    - Constructs a preprocessing pipeline for both numerical and categorical features.
    - Handles missing values, scaling, and encoding in a systematic way.
    - Reads raw train and test data, applies transformations, and outputs processed arrays ready for modeling.
    - Uses logging for traceability and custom exceptions for robust error handling.

Order of operations:
1. Import all required libraries and modules.
2. Define configuration for saving the preprocessor object.
3. Build transformation pipelines for numerical and categorical data.
4. Read and split the data into features and target.
5. Apply the preprocessing pipelines to both train and test data.
6. Return the processed data and preprocessor path for downstream tasks.

This structure ensures that all data is consistently and correctly prepared for machine learning workflows,
improving model performance and reproducibility.
"""