# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the local mlruns directory into the container
# IMPORTANT: This assumes your mlruns directory is in the same directory as the Dockerfile
# Adjust the source path if necessary.
COPY mlruns ./mlruns

# Copy the FastAPI application code into the container
COPY main.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for MLflow tracking URI inside the container
# Points to the copied mlruns directory within the container
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Define the command to run the application using uvicorn
# Use 0.0.0.0 to allow access from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 