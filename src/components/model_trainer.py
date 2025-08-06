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
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.custom_logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_model(self, model, param_grid, X_train, y_train):
        try:
            grid = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                scoring='r2',
                                n_jobs=-1,
                                cv=3,
                                verbose=1)
            grid.fit(X_train, y_train)
            return grid.best_estimator_
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
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                },
                "XGBoost": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },
                "CatBoost": {
                    'iterations': [100, 200],
                    'depth': [4, 6],
                    'learning_rate': [0.01, 0.1]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1]
                }
            }

            model_report = {}
            best_model = None
            best_score = float('-inf')

            for name, model in models.items():
                logging.info(f"Starting tuning for {name}")
                param_grid = param_grids.get(name, {})

                tuned_model = self.tune_model(model, param_grid, X_train, y_train)

                # Evaluate train performance
                y_train_pred = tuned_model.predict(X_train)
                train_r2 = r2_score(y_train, y_train_pred)

                # Evaluate test performance
                y_test_pred = tuned_model.predict(X_test)
                test_r2 = r2_score(y_test, y_test_pred)
                mse = mean_squared_error(y_test, y_test_pred)

                # Cross-validation check
                cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=3)
                cv_mean = cv_scores.mean()

                logging.info(f"{name} Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, CV R²: {cv_mean:.4f}")

                model_report[name] = {
                    "Train R2": round(train_r2, 4),
                    "Test R2": round(test_r2, 4),
                    "CrossVal R2": round(cv_mean, 4),
                    "Mean Squared Error": round(mse, 4),
                    "Model": tuned_model
                }

                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = tuned_model

            if best_score < 0.6:
                raise CustomException("No model passed the minimum acceptable R² score of 0.6", sys)

            # Save the best model
            logging.info(f"Saving best model to: {self.model_trainer_config.trained_model_file_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training and selection complete.")
            return best_model, model_report

        except Exception as e:
            raise CustomException(e, sys)
            logging.error(f"Error in model training: {str(e)}")