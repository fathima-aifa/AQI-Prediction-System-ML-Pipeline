import yaml

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def read_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def evaluate_model(y_true, y_pred):
    """
    Returns MAE, RMSE and R2 score.
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2