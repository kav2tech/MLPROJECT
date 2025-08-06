import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.custom_logger import logging  # lowercase for Docker

def save_object(file_path: str, obj: object):
    """
    Save a Python object to the given file path with debugging.
    """
    try:
        logging.debug(f"[save_object] Target path: {file_path}")
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.debug(f"[save_object] Directory ensured: {dir_path}")

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

        logging.info(f"[save_object] Object saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"[save_object] Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Evaluate multiple models with GridSearchCV and log details.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"[evaluate_model] Starting GridSearchCV for: {model_name}")
            param_grid = params.get(model_name, {})
            logging.debug(f"[evaluate_model] Param grid for {model_name}: {param_grid}")

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='r2',
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            logging.debug(f"[evaluate_model] GridSearchCV complete for {model_name}")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model_report[model_name] = {
                "Best Params": grid_search.best_params_,
                "Mean Squared Error": round(mse, 4),
                "R2 Score": round(r2, 4),
                "Model": best_model
            }

            logging.info(f"[evaluate_model] {model_name} Best Params: {grid_search.best_params_}")
            logging.info(f"[evaluate_model] {model_name} MSE: {mse:.4f}, R2: {r2:.4f}")

        return model_report

    except Exception as e:
        logging.error(f"[evaluate_model] Error evaluating models: {e}")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a Python object from the given file path with debugging.
    """
    try:
        logging.debug(f"[load_object] Attempting to load object from: {file_path}")

        if not os.path.exists(file_path):
            logging.error(f"[load_object] File not found: {file_path}")
            raise CustomException(f"File not found: {file_path}", sys)

        with open(file_path, 'rb') as file:
            obj = dill.load(file)

        logging.info(f"[load_object] Object loaded successfully from {file_path}")
        return obj

    except FileNotFoundError as fnf:
        logging.error(f"[load_object] File not found exception: {file_path}")
        raise CustomException(f"File not found: {file_path}", sys) from fnf
    except Exception as e:
        logging.error(f"[load_object] Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)



