import os
import sys
import numpy as np
import pandas as pd
from src.Exception import CustomException

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