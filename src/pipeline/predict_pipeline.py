import joblib
import pandas as pd

from datetime import datetime

from src.utils import read_yaml


class PredictPipeline:

    def __init__(self):

        config = read_yaml("config/config.yaml")

        self.model_path = config["model_trainer"]["model_path"]
        self.preprocessor_path = config["data_transformation"]["preprocessor_path"]

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, features):

        data_scaled = self.preprocessor.transform(features)

        prediction = self.model.predict(data_scaled)

        return prediction

class CustomData:

    def __init__(
        self,
        location,
        condition_text,
        temp_c,
        humidity,
        pressure_mb,
        windspeed_kph,
        pm2_5,
        pm10,
        co,
        no2
    ):

        self.location = location
        self.condition_text = condition_text
        self.temp_c = temp_c
        self.humidity = humidity
        self.pressure_mb = pressure_mb
        self.windspeed_kph = windspeed_kph
        self.pm2_5 = pm2_5
        self.pm10 = pm10
        self.co = co
        self.no2 = no2
    
    def get_data_as_dataframe(self):

        current_hour = datetime.now().hour
        current_month = datetime.now().month

        custom_data_input = {

            "location": [self.location],
            "condition_text": [self.condition_text],
            "temp_c": [self.temp_c],
            "humidity": [self.humidity],
            "pressure_mb": [self.pressure_mb],
            "windspeed_kph": [self.windspeed_kph],
            "pm2_5": [self.pm2_5],
            "pm10": [self.pm10],
            "co": [self.co],
            "no2": [self.no2],
            "hour": [current_hour],
            "month": [current_month]

        }

        return pd.DataFrame(custom_data_input)