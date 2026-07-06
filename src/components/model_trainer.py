import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from src.components.data_transformation import DataTransformation
from src.utils import read_yaml, evaluate_model


class ModelTrainer:

    def __init__(self):

        self.params = read_yaml("params.yaml")
        self.config = read_yaml("config/config.yaml")
        self.model_path = self.config["model_trainer"]["model_path"]


    def initiate_model_training(self):

        print("Starting Model Training...")

        transformer = DataTransformation()

        X_train, X_test, y_train, y_test, _ = (
            transformer.initiate_data_transformation()
        )

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

            prediction = best_model.predict(X_test)
            
            mae, rmse, r2 = evaluate_model(
                 y_test,
                 prediction
            )
            

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

        print("\nModel saved successfully.")

        return self.model_path


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.initiate_model_training()