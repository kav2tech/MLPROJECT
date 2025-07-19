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
from src.utils import save_object
from model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split   
from src.utils import evaluate_model 

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models)

            for name, model in models.items():
                logging.info(f"Training model: {name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                model_report[name] = {
                    "Mean Squared Error": round(mse, 4),
                    "R2 Score": round(r2, 4)
                }

                logging.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

            # Select best model based on R2 score
            best_model_score= max(model_report, key=lambda x: model_report[x]['R2 Score'])
            best_model = models[best_model_score]

            logging.info(f"Best model: {best_model_score} with R2 Score: {model_report[best_model_score]['R2 Score']:.4f}")

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
            logging.info("Saving the best model to disk")
            logging.info(f"Best model: {best_model_score} with R2 Score: {model_report[best_model_score]['R2 Score']:.4f}")

            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            logging.info("Model training completed successfully")
            logging.info(f"Model report: {model_report}")

            return best_model, model_report

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)



