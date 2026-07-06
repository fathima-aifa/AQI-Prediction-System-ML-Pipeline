#  AQI Prediction System (ML + MLOps Pipeline)

## 📌 Overview
This project is an end-to-end Machine Learning system that predicts the **Air Quality Index (AQI)** based on atmospheric and environmental conditions. It features a complete ML pipeline from data ingestion to evaluation, deployed as an interactive web application.

---

##  Problem Statement
Air pollution significantly impacts public health and environmental safety. The goal of this project is to:
> Predict real-time AQI values using environmental features like temperature, humidity, pressure, wind speed, and pollutant concentrations ($PM_{2.5}$, $PM_{10}$, $CO$, $NO_2$).

---

##  Machine Learning Models Used
The following regression models were trained, tuned, and compared:
* Linear Regression
* Decision Tree Regressor
* XGBoost Regressor
* **Random Forest Regressor** ⭐ *(Selected Best Model)*

###  Best Model Performance
The **Random Forest Regressor** achieved the highest accuracy across all metrics:
* **$R^2$ Score:** ~0.92
* **Mean Absolute Error (MAE):** ~35
* **Root Mean Squared Error (RMSE):** ~91

---

##  MLOps Integration (MLflow & DagsHub)
* **MLflow:** Used for experiment tracking, logging hyperparameter runs, and tracking performance metrics ($MAE$, $RMSE$, $R^2$) to easily find the best model.
* **DagsHub:** Acts as our centralized remote platform, hosting our remote MLflow tracking server and storing versioned models and pipeline artifacts.

---

##  Tech Stack
* **Core Language:** Python(version: 3.10) 
* **Data & Machine Learning:** Pandas, NumPy, Scikit-learn, XGBoost, Joblib
* **Experiment Tracking & MLOps:** MLflow, DVC, DagsHub
* **Web Deployment:** Streamlit

---

## 🚀 How to Run the Project

###  Setup & Dependencies
Clone the repository and install all required libraries:
```bash
git clone [https://github.com/fathima-aifa/AQI-Prediction-System.git](https://github.com/fathima-aifa/AQI-Prediction-System.git)
cd AQI-Prediction-System
pip install -r requirements.txt
```

###  Train the Pipeline
Run the model trainer component to process the data and save the best model:
```bash
python -m src.components.model_trainer
```

###  Launch the App
Run the Streamlit interface locally:
```bash
streamlit run app.py
```
