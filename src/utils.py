import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.Exception import CustomException
from src.Custom_logger import logging
import dill

def save_object(file_path: str, obj: object):
    """
    Saves the given object to the specified file path using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)   
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved successfully to {file_path}")
    except Exception as e:  
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys) from e

def evaluate_model(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Evaluates models using GridSearchCV and returns a report containing
    best parameters, R2 Score, and MSE for each model.
    """
    try:
        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Performing GridSearchCV for {model_name}")

            param_grid = params.get(model_name, {})

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='r2',
                verbose=1
            )
            grid_search.fit(X_train, y_train)

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

            logging.info(f"{model_name} - Best Params: {grid_search.best_params_}")
            logging.info(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}")
           

        return model_report

    except Exception as e:  
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys) from e
