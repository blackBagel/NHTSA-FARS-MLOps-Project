FROM python:3.11.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "code/model_training", "./" ]

ENV PROJECT_PATH="./"
ENV MLFLOW_EXPERIMENT_ID=1
ENV MODEL_ARTIFACT_PATH="model"
ENV MODEL_LABELS_DICT_FILE="NHTSA_FARS_labels_for_target.json"
ENV MODELS_DATASETS_PATH="/datasets"
ENV TRAIN_FILENAME='train.csv'
ENV VALIDATION_FILENAME='validation.csv'
ENV TEST_FILENAME='test.csv'
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"
ENV PREFECT_API_URL="http://prefect:4200/api"
ENV CHAMPION_MODEL_NAME="starter_notebook_model"
ENV CHAMPION_MODEL_ALIAS="Champion"

ENTRYPOINT [ "python", "entrypoint.py" ]