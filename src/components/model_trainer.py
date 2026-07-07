import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI")
)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from src.utils import read_yaml, evaluate_model


class ModelTrainer:

    def __init__(self):

        self.params = read_yaml("params.yaml")
        self.config = read_yaml("config/config.yaml")

        self.model_path = self.config["model_trainer"]["model_path"]

        self.x_train_path = self.config["data_transformation"]["x_train_path"]
        self.x_test_path = self.config["data_transformation"]["x_test_path"]
        self.y_train_path = self.config["data_transformation"]["y_train_path"]
        self.y_test_path = self.config["data_transformation"]["y_test_path"]


    def initiate_model_training(self):

        print("Starting Model Training...")
        
        mlflow.set_experiment("AQI Prediction")

        with mlflow.start_run():

            print("Loading transformed data...")

            X_train = np.load(self.x_train_path, allow_pickle=True)
            X_test = np.load(self.x_test_path, allow_pickle=True)
            y_train = np.load(self.y_train_path, allow_pickle=True)
            y_test = np.load(self.y_test_path, allow_pickle=True)
            
            # -----------------------------------
            # Models
            # -----------------------------------

            models = {

                "Linear Regression":
                    LinearRegression(),

                "Decision Tree":
                    DecisionTreeRegressor(
                        random_state=self.params["model"]["random_state"]
                    ),

                "Random Forest":
                    RandomForestRegressor(
                        random_state=self.params["model"]["random_state"]
                    ),

                "XGBoost":
                    XGBRegressor(
                        random_state=self.params["model"]["random_state"]
                    )

            }

            results = {}

            print("\nTraining Models...\n")

            for name, model in models.items():

                model.fit(X_train, y_train)

                prediction = model.predict(X_test)

                mae, rmse, r2 = evaluate_model(
                    y_test,
                    prediction
                )

                mlflow.log_metric(f"{name}_MAE", mae)
                mlflow.log_metric(f"{name}_RMSE", rmse)
                mlflow.log_metric(f"{name}_R2", r2)

                results[name] = {

                    "model": model,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2

                }

                print(f"{name}")
                print(f"MAE : {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R2  : {r2:.4f}")
                print("-" * 40)

            # -----------------------------------
            # Best Model
            # -----------------------------------

            best_model_name = max(
                results,
                key=lambda x: results[x]["R2"]
            )

            print(f"\nBest Model: {best_model_name}")

            mlflow.log_param("best_model", best_model_name)

            # -----------------------------------
            # Hyperparameter Tuning
            # -----------------------------------

            if best_model_name == "Random Forest":

                print("\nPerforming Hyperparameter Tuning...")

                rf = RandomForestRegressor(
                    random_state=self.params["model"]["random_state"]
                )

                grid = GridSearchCV(

                    estimator=rf,

                    param_grid=self.params["grid_search"],

                    cv=self.params["training"]["cv"],

                    scoring=self.params["training"]["scoring"],

                    n_jobs=-1

                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                print("\nBest Parameters")

                print(grid.best_params_)

                mlflow.log_params(grid.best_params_)

                prediction = best_model.predict(X_test)
                
                mae, rmse, r2 = evaluate_model(
                    y_test,
                    prediction
                )

                mlflow.log_metric("Tuned_RF_MAE", mae)
                mlflow.log_metric("Tuned_RF_RMSE", rmse)
                mlflow.log_metric("Tuned_RF_R2", r2)

                print("\nTuned Random Forest")

                print(f"MAE : {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R2  : {r2:.4f}")

            else:

                best_model = results[best_model_name]["model"]

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            joblib.dump(
                best_model,
                self.model_path
            )

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model"
            )

            print("\nModel saved successfully.")

            return self.model_path


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.initiate_model_training()