import os
import sys
import numpy as np
import pandas as pd
from src.Exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def save_object(file_path: str, obj: object):
    """
    Saves the given object to the specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)   
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            import dill
            dill.dump(obj, file)
    except Exception as e:  
        print(f"Error saving object: {e}")
        raise CustomException(e, sys) from e
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluates the given models on provided data and returns a report with MSE and R2 Score.
    """
    try:
        model_report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model_report[name] = {
                "Mean Squared Error": round(mse, 4),
                "R2 Score": round(r2, 4)
            }

        return model_report
        
    
    except Exception as e:  
        print(f"Error evaluating models: {e}")
        raise CustomException(e, sys) from e
        logging.info("Data Ingestion method starts")
        return model_report