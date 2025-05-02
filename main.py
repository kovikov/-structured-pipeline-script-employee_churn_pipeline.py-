#!/usr/bin/env python
# coding: utf-8

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import uvicorn
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# --- Configuration ---
# Use environment variables or default values
# Assumes MLflow tracking server is running locally or accessible
# For local default tracking URI: file:///path/to/your/mlruns
# Ensure MLFLOW_TRACKING_URI is set if not using local default

# Re-enable reading from environment variable set in Dockerfile or default to container path
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
print(f"Using MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}") # Add print for debugging
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) # Re-enable setting the URI

REGISTERED_MODEL_NAME = "EmployeeChurnModel"
MODEL_VERSION = "2" # Based on the last successful registration

# --- Load Artifacts --- 
scaler = None
model = None
feature_names = None
expected_columns_after_encoding = [] # Will be populated from feature_names.json
original_categorical_features = [] # Will be populated based on feature_names structure


def load_artifacts(model_name, version):
    global scaler, model, feature_names, expected_columns_after_encoding, original_categorical_features
    try:
        print(f"Loading artifacts for model '{model_name}' version {version}...")
        # Construct model URI for registry
        model_uri = f"models:/{model_name}/{version}"
        run_uri = mlflow.pyfunc.load_model(model_uri).metadata.run_id
        print(f"Associated Run ID: {run_uri}")

        # Load scaler
        scaler_uri = f"runs:/{run_uri}/scaler"
        scaler = mlflow.sklearn.load_model(scaler_uri)
        print("Scaler loaded successfully.")

        # Load feature names
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_uri, "feature_names.json")
        with open(local_path, 'r') as f:
            features_dict = json.load(f)
        expected_columns_after_encoding = features_dict['feature_names']
        print("Feature names loaded successfully.")
        # Infer original categorical features from the difference in names
        # This is a heuristic - assumes feature names follow pandas get_dummies format
        # Example: 'SALARY_low', 'SALARY_medium' imply 'SALARY' was original
        potential_original_cats = set()
        for col in expected_columns_after_encoding:
            if '_' in col:
                 # Check common pandas prefixes (could be improved)
                 parts = col.split('_')
                 # Simple check: if the part before the last underscore isn't numeric/float-like
                 try:
                    float(parts[-1]) # Check if last part is numeric
                    prefix = "_".join(parts[:-1])
                 except ValueError:
                     prefix = col # If last part isn't number, maybe it's the value like 'SALARY_high'
                     if len(parts) > 1:
                         prefix = "_".join(parts[:-1])

                 # Avoid adding parts of feature names like LAST_EVALUATION
                 # A better approach would be to explicitly log original cat features
                 # For now, we manually define them based on our known data
                 # potential_original_cats.add(prefix)

        # Manually define based on the known columns from the EDA/pipeline script
        original_categorical_features = ['DEPARTMENTS', 'SALARY', 'JOB_ROLE']
        print(f"Inferred original categorical features: {original_categorical_features}")

        # Load model
        model_uri_artifact = f"runs:/{run_uri}/model"
        model = mlflow.sklearn.load_model(model_uri_artifact)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # Depending on deployment strategy, might want to exit or raise
        scaler = None
        model = None
        feature_names = None

# --- API Definition ---
app = FastAPI(title="Employee Churn Prediction API", version="1.0")

# Define input data model using Pydantic
# Should match the *original* features the user provides (before encoding/scaling)
# Exclude target variable (QUIT_THE_COMPANY) and identifiers (EMPLOYEE_ID)
class EmployeeFeatures(BaseModel):
    SATISFACTION_LEVEL: float = Field(..., example=0.75)
    LAST_EVALUATION: float = Field(..., example=0.8)
    NUMBER_PROJECT: float = Field(..., example=4.0)
    AVERAGE_MONTLY_HOURS: int = Field(..., example=200)
    TIME_SPEND_COMPANY: float = Field(..., example=3.0)
    WORK_ACCIDENT: float = Field(..., example=0.0)
    PROMOTION_LAST_5YEARS: int = Field(..., example=0)
    DEPARTMENTS: str = Field(..., example="sales")
    SALARY: str = Field(..., example="medium")
    ABSENTEEISM: int = Field(..., example=5)
    JOB_ROLE: str = Field(..., example="Sales Executive")
    MANAGER_FEEDBACK_SCORE: float = Field(..., example=7.5)
    REMOTE_WORK: int = Field(..., example=1)
    ENGAGEMENT_SCORE: float = Field(..., example=0.8)

# Define output data model
class PredictionOut(BaseModel):
    prediction: int # 0 for Stayed, 1 for Quit
    # probability: float # Optional: add if model provides probabilities

@app.on_event("startup")
async def startup_event():
    """Load artifacts when the application starts."""
    load_artifacts(REGISTERED_MODEL_NAME, MODEL_VERSION)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Churn Prediction API. Use the /predict endpoint."}

@app.post("/predict", response_model=PredictionOut)
def predict_churn(employee_data: EmployeeFeatures):
    """Predicts employee churn based on input features."""
    if not model or not scaler or not expected_columns_after_encoding:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded properly.")

    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([employee_data.dict()])

        # Preprocessing:
        # 1. One-Hot Encode categorical features
        input_encoded = pd.get_dummies(input_df, columns=original_categorical_features, drop_first=True, dtype=int)

        # 2. Align columns with the training data (add missing columns with 0, remove extra)
        input_aligned = input_encoded.reindex(columns=expected_columns_after_encoding, fill_value=0)

        # 3. Scale features using the loaded scaler
        input_scaled = scaler.transform(input_aligned)

        # Make prediction
        prediction = model.predict(input_scaled)
        result = int(prediction[0]) # Get the first prediction as integer

        return {"prediction": result}

    except Exception as e:
        print(f"Prediction error: {e}") # Log error for debugging
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

# --- Run API (for local testing) ---
if __name__ == "__main__":
    # Load artifacts immediately for local testing
    load_artifacts(REGISTERED_MODEL_NAME, MODEL_VERSION)
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000) 