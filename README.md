#  AQI Prediction System (ML + MLOps Pipeline)

#### A reproducible end-to-end machine learning pipeline built using DVC, MLflow, and DagsHub.

##  Overview

his repository extends my original **AQI Prediction System** by transforming it into a reproducible end-to-end MLOps workflow.

While the original project focused on data analysis, model development, evaluation, and deployment using Streamlit, this repository focuses on automating and managing the complete machine learning lifecycle.

The project introduces modular pipeline components, experiment tracking, dataset and artifact versioning, and reproducible execution using modern MLOps tools such as **DVC**, **MLflow**, and **DagsHub**.


---

##  Problem Statement
Air pollution significantly impacts public health and environmental safety. The goal of this project is to:
> Predict real-time AQI values using environmental features like temperature, humidity, pressure, wind speed, and pollutant concentrations ($PM_{2.5}$, $PM_{10}$, $CO$, $NO_2$).



##  Objectives
This project demonstrates how to:
* build modular ML pipelines
* automate preprocessing and model training
* version datasets and artifacts
* track experiments
* reproduce results consistently
* manage machine learning workflows
  


##  Tech Stack
* **Core Language:** Python(version: 3.10) 
* **Data & Machine Learning:** Pandas, NumPy, Scikit-learn, XGBoost, Joblib
* **Experiment Tracking & MLOps:** MLflow, DVC, DagsHub
* **Environment Management:** Conda
* **Web Deployment:** Streamlit

---
## Repository  Structure

```text
AQI-Prediction-System-ML-Pipeline
│
├── notebooks/
│   ├── eda_analysis.ipynb       → Exploratory Data Analysis (EDA) and data visualization.
│   └── model_building.ipynb     → Feature engineering, model training, evaluation, and model selection.
│
├── src/
│   ├── components/              → Modular pipeline components for data ingestion, transformation, and model training.
│   ├── pipeline/                → Prediction and inference pipeline.
│   └── utils.py                 → Utility functions used across the project.
│
├── config/                      → Configuration files for the pipeline.
├── data/                        → Raw and processed datasets managed with DVC.
├── artifacts/                   → Saved preprocessing objects and trained models.
│
├── dvc.yaml                     → Defines the complete DVC pipeline workflow.
├── dvc.lock                     → Records pipeline stages and tracked outputs.
├── params.yaml                  → Hyperparameter configuration.
├── app.py                       → Streamlit application for AQI prediction.
├── requirements.txt             → Project dependencies.
└── README.md                    → Project documentation.
```
>**Note**
> The notebooks document the initial development phase of the project, including exploratory data analysis, feature engineering, model comparison, and evaluation. Once the workflow was finalized, it was converted into a modular and reproducible machine learning pipeline using DVC and MLflow.

## Models Used

The following regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

The best-performing model was selected based on evaluation metrics and further optimized using GridSearchCV for hyperparameter tuning.

## 🔄 MLOps Workflow

To improve reproducibility and experiment management, this project integrates **MLflow** and **DVC**.

Together they provide:

- End-to-end experiment tracking and model comparison.
- Version control for datasets, models, and pipeline artifacts.
- A reproducible machine learning pipeline that can be executed using `dvc repro`.

### View on DagsHub

DVC Pipeline & Repository 

https://dagshub.com/fathimathulaifa03/AQI-Prediction-System

---
## How to Run

### Clone the repository

```bash
git clone <repository-url>
cd AQI-Prediction-System-ML-Pipeline
```

### Create a virtual environment

```bash
conda create -n aqi python=3.11
conda activate aqi
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Execute the pipeline

```bash
dvc repro
```

### Launch the application

```bash
streamlit run app.py
```



## Future Enhancements

Possible improvements include:

- Automate model retraining when new data becomes available.
- Integrate CI/CD for automated testing and deployment.
- Containerize the application using Docker.
- Deploy the pipeline on cloud platforms.
- Add data validation before model training.
- Monitor model performance and data drift in production.



