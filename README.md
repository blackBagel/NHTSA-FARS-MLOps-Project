# Car Accident Injury Prediction

## Project Overview

This project aims to predict the injury category a person might sustain in a car accident using a machine learning model. By analyzing various factors such as the person's age, gender, seating position, and use of safety devices, the model classifies the severity of injuries that could occur. The prediction model is trained on data provided by the US National Highway Traffic Safety Administration (NHTSA) Fatality Analysis Reporting System (FARS). This project can help in understanding and mitigating injury risks in motor vehicle accidents.

## Problem Statement

Motor vehicle accidents are a leading cause of injury and death globally. The severity of injuries sustained in these accidents can vary widely based on numerous factors such as the individual's seating position, the use of safety devices like seatbelts or airbags, and demographic characteristics like age and gender. However, predicting the likelihood and severity of injuries remains a challenge. The objective of this project is to create a predictive model that can accurately categorize the type of injury a person might sustain in a car accident. This model can be a crucial tool for automotive safety research, policy-making, and personalized vehicle safety design for developing strategies to reduce the impact of motor vehicle accidents.

## Data Description

The model is built using the "Person" data file from the FARS database, which includes detailed records of individuals involved in motor vehicle accidents. 
You can read more about the project and the data at the foloowing links:
  - [Project Home](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars)
  - [Data Manual](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813556)
  - [File Server](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/)

There's no need to manually download data files. I use a download script in the project setup as described in the "Running the Project" section

The Person.csv data covers both motorists and non-motorists and includes the following key features:

- **Age**: The age of the individual involved in the accident.
- **Sex/Gender**: The gender of the individual.
- **Person Type**: Classification of the person as a driver, passenger, pedestrian, etc.
- **Injury Severity**: The severity of the injury sustained, ranging from no injury to fatal injury.
- **Seating Position**: The seat occupied by the individual in the vehicle at the time of the accident.
- **Restraint System Use**: Whether a seatbelt or similar restraint was used.
- **Indication of Restraint System Misuse**: Any signs of improper use of the restraint system.
- **Helmet Use**: Whether a helmet was used (relevant for motorcyclists and certain other vehicle users).
- **Indication of Helmet Misuse**: Any signs of improper use of the helmet.
- **Air Bag Deployed**: Whether the airbag was deployed during the accident.
- **Ejection**: Whether the person was ejected from the vehicle during the accident.
- **Ejection Path**: The path taken if the person was ejected from the vehicle.

Each record in the dataset is uniquely identified using a combination of **ST_CASE**, **STATE**, **VEH_NO**, and **PER_NO**.

## How the Model Works

The model is a classification model that uses the features described above to predict the injury category (such as no injury, minor injury, severe injury, or fatal injury) for each person involved in a vehicle accident. The training process involves feeding the model with historical data, enabling it to learn patterns and correlations between the input features and the injury outcomes.

## Getting Started

To get started with this project, you need to first set up a new project on Google Cloud Platform.
You can follow the instructions at [GCP Setup](GCP%20setup.md) to set up everything you'll need in your GCP project.
Once you've set up your VM, service accounts and storage bucket you can SSH into your machine, and clone this repository.

### VM Setup

#### 1. Install Docker Engine, Docker Compose, and Prerequisites

1. **Install Docker Engine**:
   - Update your package manager:
     ```bash
     sudo apt-get update
     ```
   - Install required packages:
     ```bash
     sudo apt-get install \
         ca-certificates \
         curl \
         gnupg \
         lsb-release
     ```
   - Add Docker’s official GPG key:
     ```bash
     sudo mkdir -m 0755 -p /etc/apt/keyrings
     curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
     ```
   - Set up the Docker repository:
     ```bash
     echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
     ```
   - Install Docker Engine:
     ```bash
     sudo apt-get update
     sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
     ```
   - Verify the Docker installation:
     ```bash
     sudo docker --version
     ```

2. **Install Docker Compose**:
   - Download the latest version of Docker Compose:
     ```bash
     sudo curl -L "https://github.com/docker/compose/releases/download/v2.22.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
     ```
   - Apply executable permissions to the binary:
     ```bash
     sudo chmod +x /usr/local/bin/docker-compose
     ```
   - Verify the Docker Compose installation:
     ```bash
     docker-compose --version
     ```

3. **Additional Prerequisites**:
   - Add your user to the Docker group so you can run Docker commands without `sudo`:
     ```bash
     sudo usermod -aG docker $USER
     ```
     Log out and log back in to apply this change.
   - Ensure your firewall allows Docker to function correctly, especially if you plan on exposing ports.

### Deploying the project

1. **Fetch the project from github**:
   ```bash
    git clone https://github.com/blackBagel/NHTSA-FARS-MLOps-Project.git
    cd NHTSA-FARS-MLOps-Project
    ```

2. **Run the `download_data.sh` script to create a data directory and get the data i used for this project**:
    ```bash
    chmod u+x ./download_data.sh
    ./download_data.sh
    ```

3. **Set the bucket path**:
    Go to the dockerfile for mlflow and set the default artifact root parameter to the path of your storage bucket:
    ```dockerfile
    CMD [ \
            "mlflow", "server", \
            "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
            "--default-artifact-root", "gs://PATH_TO_YOUR_BUCKET", \
            "--host", "0.0.0.0", \
            "--port", "5000" \
        ]
    ```

4. **Copy your service account access keys**
    - Don't forget to give your files meaningful names!
    ```
    mkdir -p secrets
    ```
    - Copy the access key files for your service accounts into this new directory
    - Change the docker-compose.yaml environment variables pointing to your json files for the following services:
        - `mlflow`
        - `accident-injury-prediction-service`

5. **Start**:
    Now you're ready to start up the project! Deploy all the containers using `docker-compose`
    ```bash
    docker-compose up
    ```

6. **Port forward**
    - Port forward the following port:
        - 5000 - Mlflow
        - 4200 - Prefect Server
    - You can deploy and run the different project components using the prefect UI at: `http://localhost:4200`
    - And see your runs at the mlflow ui: `http://localhost:5000`

## Project Workflow

The ML pipeline of the project is run using prefect scheduled deployments.
By default the deployments run weekly since I felt this is a decent time for data accumulation that is worth retraining.

### Container Services Overview

#### 1. **mlflow**
   - **Purpose:** Manages the tracking and registry of machine learning experiments.
   - **Ports:** Exposes port `5000` for accessing the MLflow tracking UI.
   - **Volumes:** 
     - `./data/mlflow:/home/mlflow/`: Stores MLflow data such as experiment logs and models.
     - `./secrets/mlflow:/run/secrets:ro`: Provides secrets, including credentials for Google Cloud.
   - **Environment:** Utilizes a Google service account for authentication with cloud services.
   - **Networks:** Connected to the `app-network`.
   - **Restart Policy:** Always restarts to ensure availability.

#### 2. **prefect**
   - **Purpose:** Orchestrates workflows and manages the execution of data pipelines.
   - **Ports:** Exposes port `4200` for accessing the Prefect UI.
   - **Volumes:** 
     - `./data/prefect:/database/`: Stores Prefect's state and configuration data.
   - **Networks:** Connected to the `app-network`.
   - **Restart Policy:** Always restarts to maintain service continuity.

#### 3. **accident-injury-datasets-updater**
   - **Purpose:** Updates and processes datasets related to accident and injury statistics.
   - **Volumes:** 
     - `./data/datasets/:/datasets`: Stores the datasets that are updated by the service.
   - **Networks:** Connected to the `app-network`.
   - **Restart Policy:** Always restarts to keep the data up-to-date.

#### 4. **accident-injury-model-trainer**
   - **Purpose:** Trains machine learning models using the processed datasets.
   - **Volumes:** 
     - `./data/datasets/for_models:/datasets:ro`: Provides read-only access to datasets for model training.
     - `./served_model_env_vars:/served_model_env_vars`: Stores environment variables for model serving.
   - **Networks:** Connected to the `app-network`.
   - **Restart Policy:** Always restarts to ensure continuous model training.

#### 5. **accident-injury-prediction-service**
   - **Purpose:** Serves the trained machine learning models for making predictions.
   - **Ports:** Exposes port `9696` for accessing the prediction service API.
   - **Volumes:** 
     - `./secrets/prediction_server:/run/secrets:ro`: Provides secrets, including credentials for Google Cloud.
   - **Environment:** Uses environment variables defined in `served_model_env_vars` for configuration.
   - **Restart Policy:** Always restarts to maintain service availability.

### Network Configuration

#### **app-network**
   - **Driver:** Uses the `bridge` driver for container networking, allowing services to communicate with each other securely within the Docker environment.

### Prefect Deployments:

- #### model_datasets_updater
  Creates new train, validation and test data every week.

  The purpose of this deployment is to simulate a live data stream using our static FARS data files

- #### candidate_models_train_deployment
  Trains many different sklearn pipelines with different models and parameters on the training data.

  The purpose of this deployment is to check whether our champion model's performance deteriorates compared to other models

- #### champion_model_retrain_deployment
  Retrains the champion model every week on the most up to date data, monitors its' performance and creates a challenger model in case the champion's performance on the training data deteriorates

  The purpose of this deployment is to keep the challenger model up to date and monitor it regularly

### Model Evaluation

The evaluation metric we'll use for our model will be a weighted average of recall per class.

The more severe an injury gets, the more important it is to decrease the amount of False Negative predictions of it, since the price of an error becomes more severe. 
Therefore, it makes sense to calculate the recall score of each injury class separately and then calculate an overall weighted average that gives higher importance to more severe injuries.

For the sake of this exercise, we'll focus only on this metric as the only maximising metric and not take into account other satisfising factors like the prediction speed of our model