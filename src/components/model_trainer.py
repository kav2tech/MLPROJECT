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
from sklearn.model_selection import RandomizedSearchCV

from src.Exception import CustomException
from src.Custom_logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_model(self, model, param_grid, X_train, y_train):
        try:
            search = RandomizedSearchCV(model, param_distributions=param_grid,
                                        n_iter=10, scoring='r2', n_jobs=-1, cv=3, random_state=42)
            search.fit(X_train, y_train)
            return search.best_estimator_
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            param_grids = {
                "Linear Regression": {},
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                },
                "XGBoost": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                },
                "CatBoost": {
                    'iterations': [100, 200],
                    'depth': [4, 6],
                    'learning_rate': [0.01, 0.1],
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1],
                }
            }

            model_report = {}
            best_model = None
            best_score = float('-inf')

            for name, model in models.items():
                logging.info(f"Hyperparameter tuning for {name}")
                param_grid = param_grids.get(name, {})
                tuned_model = self.tune_model(model, param_grid, X_train, y_train)
                y_pred = tuned_model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                model_report[name] = {
                    "Mean Squared Error": round(mse, 4),
                    "R2 Score": round(r2, 4)
                }

                if r2 > best_score:
                    best_score = r2
                    best_model = tuned_model

                logging.info(f"{name} => MSE: {mse:.4f}, R2: {r2:.4f}")

            if best_score < 0.6:
                raise CustomException("No suitable model found with R2 Score above 0.6", sys)
            
            # Select best model based on highest R2 Score
            best_model_name = max(model_report, key=lambda name: model_report[name]["R2 Score"])
            best_model_score = model_report[best_model_name]["R2 Score"]

            logging.info(f"✅ Best model: {best_model_name} with R2 Score: {best_model_score:.4f}")
            
            # Save the best model
            logging.info(f"Saving the best model to {self.model_trainer_config.trained_model_file_path}")
            if not os.path.exists(os.path.dirname(self.model_trainer_config.trained_model_file_path)):
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path))
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model: {best_model} with R2 Score: {best_score:.4f}")
            return best_model, model_report

        except Exception as e:
            raise CustomException(e, sys)


