# Employee Churn Prediction Pipeline

This project implements an end-to-end machine learning pipeline to predict employee churn using a sample dataset. It covers data preprocessing, model training, experiment tracking with MLflow, and deployment as a REST API using FastAPI and Docker.

## Project Goal

The main objective is to build a system that can predict whether an employee is likely to leave the company (churn) based on various features related to their work, satisfaction, and demographics. This prediction can help HR departments proactively identify at-risk employees and implement retention strategies.

## Features Implemented

*   **Data Loading & Inspection:** Loads data from a CSV file and performs initial exploratory analysis.
*   **Data Preprocessing:**
    *   Handles missing values (skipped in this case as none were found).
    *   Feature Engineering (drops identifiers, one-hot encodes categorical features).
    *   Feature Scaling (using `StandardScaler`).
    *   Anomaly Detection (using IQR, detection only).
    *   Class Imbalance Handling (using SMOTE on the training data).
*   **Model Training & Evaluation:**
    *   Trains a baseline `DummyClassifier`.
    *   Trains `LogisticRegression` and `RandomForestClassifier`.
    *   Evaluates models using standard classification metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC).
*   **Experiment Tracking:** Uses MLflow to log parameters, metrics, and artifacts (model, scaler, feature names) for each model run.
*   **Model Registration:** Registers the selected best model in the MLflow Model Registry.
*   **API Deployment:**
    *   Develops a prediction API using FastAPI.
    *   The API loads the registered model and necessary preprocessing artifacts from MLflow.
    *   Provides a `/predict` endpoint to get predictions for new employee data.
*   **Containerization:** Packages the FastAPI application and its dependencies into a Docker container for easy deployment.

## Project Structure

```
BIG_PROJECT/
├── mlruns/               # MLflow tracking data (created automatically)
├── .gitignore            # Specifies intentionally untracked files
├── employee_churn_pipeline.py # Main script for data processing, training, evaluation, logging
├── main.py               # FastAPI application script for the prediction API
├── requirements.txt      # Python dependencies for the project
├── Dockerfile            # Instructions to build the Docker image for the API
├── DATA.csv              # The dataset used for training (ensure this exists)
└── README.md             # This file
```

*(Note: The `mlruns/` directory is created by MLflow when the pipeline script is run.)*

## Setup

**Prerequisites:**

*   Python (version 3.9+ recommended)
*   Git
*   Docker Desktop (or Docker Engine) installed and running.

**Steps:**

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd BIG_PROJECT
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create environment (e.g., named 'myenv')
    python -m venv myenv

    # Activate environment
    # Windows PowerShell:
    .\myenv\Scripts\Activate.ps1
    # Windows CMD:
    myenv\Scripts\activate.bat
    # macOS/Linux:
    source myenv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *You might also need `imbalanced-learn` which was installed separately:*
    ```bash
    pip install imbalanced-learn
    ```

## Running the Pipeline

This script performs all steps up to model training, evaluation, and registration in MLflow.

1.  **Ensure `DATA.csv` is present** in the `BIG_PROJECT` directory.
2.  **Run the pipeline script:**
    ```bash
    (myenv) python employee_churn_pipeline.py
    ```
3.  **View MLflow Results (Optional):**
    *   In a separate terminal (in the `BIG_PROJECT` directory):
        ```bash
        (myenv) mlflow ui
        ```
    *   Open your browser to `http://127.0.0.1:5000` to view experiments, runs, metrics, and registered models.

## Running the Prediction API

There are two ways to run the prediction API:

**Option A: Local Python Run (Requires MLflow access)**

This runs the FastAPI app directly using your local Python environment. It relies on the `mlruns` directory being accessible.

1.  **Run the FastAPI app:**
    ```bash
    (myenv) python main.py
    ```
2.  **Access the API:**
    *   Go to `http://localhost:8000/docs` in your browser for the Swagger UI to send test requests.

**Option B: Docker Run (Recommended for Deployment)**

This builds a self-contained Docker image and runs the API inside a container.

1.  **Build the Docker Image:** (Make sure Docker Desktop is running)
    ```bash
    docker build -t employee-churn-api .
    ```
2.  **Run the Docker Container:**
    ```bash
    docker run -p 8000:8000 --name churn-predictor employee-churn-api
    ```
3.  **Access the API:**
    *   Go to `http://localhost:8000/docs` in your browser for the Swagger UI to send test requests.
4.  **Stop the Container:**
    *   Press `Ctrl+C` in the terminal where `docker run` is executing, or run `docker stop churn-predictor` from another terminal.

## Future Work / Next Steps

*   Implement CI/CD pipelines using GitHub Actions or similar tools.
*   Set up automated model retraining workflows (e.g., using Airflow).
*   Write unit and integration tests for the pipeline and API code.
*   Experiment with other models (e.g., Gradient Boosting - XGBoost, LightGBM).
*   Perform more in-depth feature selection/engineering.
*   Deploy the Docker container to a cloud platform (AWS, Azure, GCP) or Kubernetes.
*   Use a remote MLflow tracking server for better collaboration and persistence. 