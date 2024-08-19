# Car Accident Injury Prediction

## Project Overview

This project aims to predict the injury category a person might sustain in a car accident using a machine learning model. By analyzing various factors such as the person's age, gender, seating position, and use of safety devices, the model classifies the severity of injuries that could occur. The prediction model is trained on data provided by the US National Highway Traffic Safety Administration (NHTSA) Fatality Analysis Reporting System (FARS). This project can help in understanding and mitigating injury risks in motor vehicle accidents.

## Problem Statement

Motor vehicle accidents are a leading cause of injury and death globally. The severity of injuries sustained in these accidents can vary widely based on numerous factors such as the individual's seating position, the use of safety devices like seatbelts or airbags, and demographic characteristics like age and gender. However, predicting the likelihood and severity of injuries remains a challenge. The objective of this project is to create a predictive model that can accurately categorize the type of injury a person might sustain in a car accident. This model can be a crucial tool for automotive safety research, policy-making, and personalized vehicle safety design for developing strategies to reduce the impact of motor vehicle accidents.

## Data Description

The model is built using the "Person" data file from the FARS database, which includes detailed records of individuals involved in motor vehicle accidents. The data covers both motorists and non-motorists and includes the following key features:

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

### Running the project

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

3. **Copy your service account access keys**
    - Don't forget to give your files meaningful names!
    ```
    mkdir -p secrets
    ```
    - Copy the access key files for your service accounts into this new directory
    - Change the docker-compose.yaml environment variables pointing to your json files for the following services:
        - `mlflow`
        - `accident-injury-prediction-service`

4. **Now you're ready to start up the project! Deploy all the containers using `docker-compose`**:
    ```bash
    docker-compose up
    ```

5. **Port forward**
    - Port forward the following port:
        - 5000 - Mlflow
        - 4200 - Prefect Server
    - You can deploy and run the different project components using the prefect UI at: `http://localhost:4200`
    - And see your runs at the mlflow ui: `http://localhost:5000`