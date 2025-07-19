import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.Exception import CustomException
from src.Custom_logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data arrays")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Support Vector Regressor": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            # Evaluate all models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Identify the best model based on R2 Score
            best_model_name = max(model_report, key=lambda x: model_report[x]["R2 Score"])
            best_model_score = model_report[best_model_name]["R2 Score"]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 Score above 0.6", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            mse = mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)
            logging.info(f"Final model evaluation - MSE: {mse:.4f}, R2: {r2:.4f}")
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return best_model, model_report

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)


