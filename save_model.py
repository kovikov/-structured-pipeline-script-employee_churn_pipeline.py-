import shutil
import os

# Delete the directory if it exists
if os.path.exists("employee_churn_model"):
    shutil.rmtree("employee_churn_model")

# Save the model
mlflow.pyfunc.save_model(path="employee_churn_model", python_model=YourModelClass())
