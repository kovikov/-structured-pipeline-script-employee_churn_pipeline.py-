#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    make_scorer
)
import mlflow
import mlflow.sklearn
import json

# Define the path to the dataset relative to the script location
# Since DATA.csv is in the same directory as the script:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "DATA.csv")

def load_data(path):
    """Loads the employee churn dataset from a CSV file."""
    print(f"Attempting to load data from: {path}")
    try:
        # Explicitly specify the encoding, common ones are utf-8 or latin1
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin1...")
            df = pd.read_csv(path, encoding='latin1')
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        print("Please ensure the data file is in the same directory as the script.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def initial_data_inspection(df):
    """Performs and prints initial data inspection."""
    if df is not None:
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info()
        print("\n--- Descriptive Statistics ---")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.describe(include='all')) # Include all columns
        print("\n--- Value Counts for Categorical Features ---")
        # Identify potential categorical columns (object type or low unique counts)
        potential_cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
             if df[col].nunique() < 20: # Arbitrary threshold for low cardinality numeric features
                 potential_cat_cols.append(col)

        for col in potential_cat_cols:
            # Check if column still exists (might be dropped)
            if col in df.columns:
                print(f"\nValue counts for {col} (Top 20):")
                print(df[col].value_counts().head(20))
            else:
                print(f"Column {col} not found for value counts.")

        print("\n--- Missing Value Counts ---")
        print(df.isnull().sum())
        print(f"\nTotal rows: {len(df)}")

    else:
        print("DataFrame is None, skipping inspection.")

def detect_outliers_iqr(df):
    """Detects outliers using the IQR method on numerical columns.

    Args:
        df (pd.DataFrame): The input DataFrame (original data).

    Returns:
        pd.Index: The index of rows containing at least one outlier.
    """
    print("\n--- Step 5: Anomaly Detection (IQR) --- ")
    # Select numerical columns for IQR calculation
    # Exclude binary/low-cardinality integer columns where IQR might not be meaningful
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_exclude = [col for col in numerical_cols if df[col].nunique() < 5] # Heuristic to exclude binary/low-cardinality
    cols_for_iqr = [col for col in numerical_cols if col not in cols_to_exclude and col != 'QUIT_THE_COMPANY'] # Also exclude target

    print(f"Columns checked for outliers using IQR: {cols_for_iqr}")

    outlier_indices = set()
    outliers_per_col = {}

    for col in cols_for_iqr:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find indices of outliers for this column
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outliers_per_col[col] = len(col_outliers)
        outlier_indices.update(col_outliers)

    num_outlier_rows = len(outlier_indices)
    total_rows = len(df)
    percentage_outliers = (num_outlier_rows / total_rows) * 100

    print("\nOutlier Counts per Column:")
    for col, count in outliers_per_col.items():
        print(f" - {col}: {count} outliers")

    print(f"\nTotal number of rows containing at least one outlier: {num_outlier_rows}")
    print(f"Percentage of rows with outliers: {percentage_outliers:.2f}%")
    print("Note: Outliers detected but not removed at this stage.")

    return df.index[list(outlier_indices)] # Return the index object

def feature_engineering(df, target_column, id_column):
    """Performs feature engineering: drops ID, separates target, one-hot encodes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.
        id_column (str): The name of the identifier column to drop.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Features (X) with categorical variables encoded.
            - pd.Series: Target variable (y).
            - list: List of columns used for features.
    """
    print(f"\n--- Step 3: Feature Engineering --- ")
    # Make a copy to avoid modifying the original DataFrame passed to the function
    df_processed = df.copy()

    # Drop the identifier column
    if id_column in df_processed.columns:
        df_processed.drop(columns=[id_column], inplace=True)
        print(f"Dropped identifier column: {id_column}")
    else:
        print(f"Identifier column {id_column} not found, skipping drop.")

    # Separate features (X) and target (y)
    if target_column in df_processed.columns:
        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])
        print(f"Separated target variable: {target_column}")
    else:
        print(f"Error: Target column {target_column} not found!")
        return None, None, None

    # Identify categorical columns (object type)
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"Identified categorical columns for encoding: {list(categorical_cols)}")

    # Apply one-hot encoding
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    print("Applied one-hot encoding.")
    print(f"Shape of X after encoding: {X_encoded.shape}")

    feature_names = X_encoded.columns.tolist()

    return X_encoded, y, feature_names

def scale_features(X):
    """Scales numerical features using StandardScaler.

    Args:
        X (pd.DataFrame): DataFrame containing features to scale.
                       Assumes all columns are numerical (after encoding).

    Returns:
        pd.DataFrame: Scaled features as a DataFrame with original column names.
        StandardScaler: The fitted scaler object.
    """
    print("\n--- Step 4: Feature Scaling --- ")
    scaler = StandardScaler()
    # Fit the scaler to the data and transform the data
    X_scaled_array = scaler.fit_transform(X)
    # Convert the scaled array back to a DataFrame with original column names
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    print("Applied StandardScaler to features.")
    print(f"Shape of scaled X: {X_scaled_df.shape}")
    return X_scaled_df, scaler

def handle_imbalance_smote(X_train, y_train, random_state=42):
    """Handles class imbalance using SMOTE on the training data.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target variable.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Resampled training features (X_train_resampled).
            - np.ndarray: Resampled training target (y_train_resampled).
            - SMOTE: The fitted SMOTE object.
    """
    print("\n--- Step 6: Handling Class Imbalance (SMOTE) --- ")
    smote = SMOTE(random_state=random_state)
    print(f"Original training set shape {Counter(y_train)}")
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training set shape {Counter(y_train_resampled)}")
    print("Applied SMOTE to the training data.")
    return X_train_resampled, y_train_resampled, smote

def train_evaluate_baseline(X_train, y_train, X_test, y_test, strategy='stratified', random_state=42):
    """Trains and evaluates a DummyClassifier baseline model.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        strategy (str): Strategy for DummyClassifier (e.g., 'stratified', 'most_frequent').
        random_state (int): Random state for reproducibility.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    print(f"\n--- Step 7: Training Baseline Model (DummyClassifier - {strategy}) ---")
    baseline_model = DummyClassifier(strategy=strategy, random_state=random_state)

    # Train on the original (potentially imbalanced) training data
    baseline_model.fit(X_train, y_train)
    print("Baseline model trained.")

    # Predict on the test set
    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1] # Probability for ROC AUC

    # Evaluate
    print("\n--- Baseline Model Evaluation (on Test Set) ---")
    accuracy = accuracy_score(y_test, y_pred_baseline)
    precision = precision_score(y_test, y_pred_baseline, zero_division=0)
    recall = recall_score(y_test, y_pred_baseline, zero_division=0)
    f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba_baseline)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    # Use target_names for better readability if desired: target_names=['Stayed', 'Quit']
    print(classification_report(y_test, y_pred_baseline, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_baseline)
    print(cm)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    return metrics

def train_evaluate_log_model(model, model_name, X_train, y_train, X_test, y_test, scaler=None, feature_names=None):
    """Trains, evaluates, and logs a model using MLflow.

    Args:
        model: Instantiated scikit-learn classifier model.
        model_name (str): Name for the model run in MLflow.
        X_train: Training features (potentially resampled).
        y_train: Training target (potentially resampled).
        X_test: Testing features.
        y_test: Testing target.
        scaler (object, optional): Fitted scaler object to log. Defaults to None.
        feature_names (list, optional): List of feature names after encoding. Defaults to None.

    Returns:
        dict: Dictionary containing evaluation metrics and run_id.
    """
    print(f"\n--- Training and Evaluating: {model_name} ---")
    with mlflow.start_run(run_name=model_name) as run:
        # Log model parameters
        try:
            params = model.get_params()
            mlflow.log_params(params)
            print(f"Logged parameters for {model_name}.")
        except Exception as e:
            print(f"Could not log parameters for {model_name}: {e}")

        # Train the model
        model.fit(X_train, y_train)
        print(f"{model_name} trained.")

        # Predict on the test set
        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except AttributeError:
            print(f"{model_name} does not support predict_proba.")
            y_pred_proba = None # Assign None if predict_proba is not available
            has_proba = False

        # Evaluate
        print(f"\n--- {model_name} Evaluation (on Test Set) ---")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        if has_proba and y_pred_proba is not None:
            try:
                 roc_auc = roc_auc_score(y_test, y_pred_proba)
                 metrics["roc_auc"] = roc_auc
                 print(f"ROC AUC: {roc_auc:.4f}")
            except ValueError as ve:
                 print(f"Could not calculate ROC AUC for {model_name}: {ve}")
                 metrics["roc_auc"] = np.nan # Log NaN if calculation fails
        else:
             metrics["roc_auc"] = np.nan # Log NaN if no probabilities

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print("Logged metrics to MLflow.")

        # Log the scaler if provided
        if scaler is not None:
            try:
                mlflow.sklearn.log_model(scaler, "scaler")
                print("Logged scaler object to MLflow.")
            except Exception as e:
                print(f"Error logging scaler: {e}")

        # Log feature names if provided
        if feature_names is not None:
            try:
                # Convert list to a dictionary format suitable for log_dict (e.g., index: name)
                # Or simply save as a JSON artifact
                features_dict = {"feature_names": feature_names}
                mlflow.log_dict(features_dict, "feature_names.json")
                print("Logged feature names to MLflow.")
            except Exception as e:
                print(f"Error logging feature names: {e}")

        # Log the model artifact (ensure this happens *after* other artifacts)
        try:
             mlflow.sklearn.log_model(model, "model") # Changed artifact path to 'model'
             print(f"Logged model '{model_name}' to MLflow artifact path 'model'.")
        except Exception as e:
            print(f"Error logging model {model_name} to MLflow: {e}")

        run_id = run.info.run_id
        print(f"MLflow Run ID for {model_name}: {run_id}")

        # Return metrics AND run_id
        results = {
            "metrics": metrics,
            "run_id": run_id
        }
    return results # Return dict containing metrics and run_id

# <<< START OF HYPERPARAMETER TUNING FUNCTION >>>
def tune_hyperparameters(model, param_dist, X_train, y_train, n_iter=10, cv=5, scoring='f1', random_state=42):
    """Performs RandomizedSearchCV for hyperparameter tuning and logs to MLflow.

    Args:
        model: The base model instance to tune (e.g., RandomForestClassifier()).
        param_dist (dict): Dictionary with parameters names (str) as keys and distributions
                           or lists of parameters to try.
        X_train: Training features (potentially resampled).
        y_train: Training target (potentially resampled).
        n_iter (int): Number of parameter settings that are sampled.
        cv (int): Number of cross-validation folds.
        scoring (str or callable): Strategy to evaluate the performance of the cross-validated model.
                                 Common options: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.
        random_state (int): Random state for reproducibility.

    Returns:
        object: The best estimator found by RandomizedSearchCV.
    """
    model_name = model.__class__.__name__
    tuning_run_name = f"{model_name}_Hyperparameter_Tuning"
    print(f"\n--- Step 10: Hyperparameter Tuning ({model_name}) --- ")
    print(f"Scoring metric: {scoring}")

    # Define the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1, # Use all available CPU cores
        verbose=1  # Print progress
    )

    # Start MLflow run for tuning
    with mlflow.start_run(run_name=tuning_run_name) as run:
        mlflow.log_param("tuning_iterations", n_iter)
        mlflow.log_param("tuning_cv_folds", cv)
        mlflow.log_param("tuning_scoring_metric", scoring)
        mlflow.log_params({"param_dist_" + k: str(v) for k, v in param_dist.items()}) # Log the search space

        print("Starting Randomized Search...")
        # Perform the search on the training data
        random_search.fit(X_train, y_train)
        print("Randomized Search completed.")

        # Log results
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        print(f"Best Parameters found: {best_params}")
        print(f"Best Cross-Validation Score ({scoring}): {best_score:.4f}")

        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric(f"best_cv_{scoring}", best_score)

        # Optionally log the best model found by the search
        # mlflow.sklearn.log_model(random_search.best_estimator_, f"{model_name}_best_tuned")

        run_id = run.info.run_id
        print(f"MLflow Run ID for Tuning ({model_name}): {run_id}")

    return random_search.best_estimator_
# <<< END OF HYPERPARAMETER TUNING FUNCTION >>>

# --- Main execution ---
if __name__ == "__main__":
    # Set MLflow Experiment
    MLFLOW_EXPERIMENT_NAME = "Employee Churn Prediction"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")

    print("--- Step 1: Data Loading & Initial Inspection ---")
    employee_df = load_data(DATA_PATH)

    if employee_df is not None:
        initial_data_inspection(employee_df)
        print("\nStep 1 completed.")

        # --- Step 5: Anomaly Detection (IQR) ---
        outlier_row_indices = detect_outliers_iqr(employee_df)
        # Currently, we only detect and report. No removal is done.
        # If removal was desired, we could do: employee_df = employee_df.drop(outlier_row_indices)
        print("\nStep 5 completed (Detection Only).")

        # --- Step 3: Feature Engineering ---
        TARGET_VARIABLE = 'QUIT_THE_COMPANY'
        ID_COLUMN = 'EMPLOYEE_ID'
        X, y, feature_names = feature_engineering(employee_df, TARGET_VARIABLE, ID_COLUMN)

        if X is not None and y is not None:
            print("\nStep 3 completed.")
            print("\nFirst 5 rows of processed features (X):")
            print(X.head())
            print(f"\nNumber of features: {len(feature_names)}")

            # --- Step 4: Feature Scaling ---
            X_scaled, scaler_object = scale_features(X)
            print("\nFirst 5 rows of scaled features (X_scaled):")
            print(X_scaled.head())
            print("\nStep 4 completed.")

            # --- Split Data into Training and Testing Sets ---
            # Split *before* applying SMOTE to avoid data leakage into test set
            print("\n--- Splitting Data (Train/Test) --- ")
            TEST_SIZE = 0.2
            RANDOM_STATE = 42
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            print(f"Data split into training ({1-TEST_SIZE:.0%}) and testing ({TEST_SIZE:.0%}) sets.")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # --- Step 6: Handle Class Imbalance (SMOTE on Training Data) ---
            X_train_res, y_train_res, smote_object = handle_imbalance_smote(X_train, y_train, random_state=RANDOM_STATE)
            print("\nStep 6 completed.")

            # --- Step 7: Train Baseline Model ---
            baseline_metrics = train_evaluate_baseline(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)
            print("\nStep 7 completed.")

            # --- Step 8 & 9: Train Models and Log with MLflow ---
            print("\n--- Step 8 & 9: Training Models & Logging --- ")
            lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
            # Capture the full results dictionary
            lr_results = train_evaluate_log_model(lr_model, "LogisticRegression", X_train_res, y_train_res, X_test, y_test)

            # Random Forest - Pass scaler and feature names for logging
            rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
            # Capture the full results dictionary for RF
            rf_results = train_evaluate_log_model(
                model=rf_model,
                model_name="RandomForestClassifier_Default",
                X_train=X_train_res,
                y_train=y_train_res,
                X_test=X_test,
                y_test=y_test,
                scaler=scaler_object,          # Pass the fitted scaler
                feature_names=feature_names    # Pass the list of feature names
            )
            # We keep the run_id of the default model in case tuning fails or is skipped
            default_rf_run_id = rf_results["run_id"]
            print("\nSteps 8 & 9 completed.")

            # --- Step 10: Hyperparameter Tuning (Random Forest) ---
            # Re-enabled
            print("\n--- Step 10: Hyperparameter Tuning (RandomForestClassifier) --- ")
            # Define parameter distribution for RandomizedSearchCV
            rf_param_dist = {
                'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            }

            # Perform tuning (using f1 score as the primary metric)
            try:
                best_rf_estimator = tune_hyperparameters(
                    model=RandomForestClassifier(random_state=RANDOM_STATE), # Pass a new instance
                    param_dist=rf_param_dist,
                    X_train=X_train_res,
                    y_train=y_train_res,
                    n_iter=50,  # Keep n_iter=50, adjust if needed for faster/slower tuning
                    cv=5,
                    scoring='f1',
                    random_state=RANDOM_STATE
                )
                print("Hyperparameter tuning completed successfully.")
                tuning_succeeded = True
            except Exception as e:
                print(f"Error during hyperparameter tuning: {e}")
                print("Falling back to default Random Forest model.")
                best_rf_estimator = rf_model # Use the default model trained earlier
                tuning_succeeded = False

            print("\nStep 10 finished.")

            # --- Step 11: Select and Evaluate Final Model ---
            print("\n--- Step 11: Select and Evaluate Final Model --- ")
            if tuning_succeeded:
                final_model = best_rf_estimator
                final_model_name = "RandomForestClassifier_Tuned"
                print(f"Selected tuned model '{final_model_name}' based on hyperparameter search.")
                # Evaluate the *tuned* model on the test set and log it
                print("\nEvaluating tuned model on test set...")
                final_model_results = train_evaluate_log_model(
                    model=final_model,
                    model_name=final_model_name,
                    X_train=X_train_res,
                    y_train=y_train_res,
                    X_test=X_test,
                    y_test=y_test,
                    scaler=scaler_object,
                    feature_names=feature_names
                )
                final_model_run_id = final_model_results["run_id"]
            else:
                final_model = rf_model # Fallback to default model
                final_model_name = "RandomForestClassifier_Default"
                final_model_run_id = default_rf_run_id # Use run_id from the default run
                print(f"Selected default model '{final_model_name}' (Run ID: {final_model_run_id}) due to tuning error.")

            print(f"Final selected model Run ID: {final_model_run_id}")
            print("Step 11 completed.")

            # --- Step 12: Save/Register Final Model ---
            print("\n--- Step 12: Register Final Model in MLflow --- ")
            MODEL_REGISTRY_NAME = "EmployeeChurnModel"
            # Use the run ID of the final selected model (either tuned or default)
            model_uri = f"runs:/{final_model_run_id}/model"
            print(f"Registering model from URI: {model_uri}")
            try:
                registered_model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=MODEL_REGISTRY_NAME
                )
                print(f"Registered model '{MODEL_REGISTRY_NAME}', Version: {registered_model_version.version}")
            except Exception as e:
                print(f"Error registering model: {e}")
            print("Step 12 completed.")

            # --- Next Steps (Placeholders) ---
            print("\n--- Chronological Steps Planned ---")
            print("1-7. Preprocessing, Baseline (Completed)")
            print("8&9. Train Models & Log with MLflow (Completed)")
            print("10. Hyperparameter Tuning (Completed)")
            print("11. Select Best Model (Completed)")
            print("12. Save Final Model (Registered in MLflow Model Registry)")
            print("13. Containerize Application (Docker)")
            print("14. Deploy Model (FastAPI)")
            print("15. Set up CI/CD & Automated Retraining")
            # ... and so on
        else:
             print("Feature engineering failed. Stopping execution.")
    else:
        print("Failed to load data. Stopping execution.") 