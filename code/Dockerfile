FROM python:3.11.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "model.py", "./" ]

ENV MLFLOW_EXPERIMENT_ID=1
ENV RUN_ID="29ea0a4121194746ad168c228c3f217d"
ENV MLFLOW_BUCKET_PATH="gs://mlops_zoomcamp-mlflow-artifacts/artifacts"
ENV MODEL_ARTIFACT_PATH="model"
ENV MODEL_LABELS_DICT_FILE="NHTSA_FARS_labels_for_target.json"
ENV GOOGLE_APPLICATION_CREDENTIALS="/run/secrets/mlops-zoomcamp-423615-33e43567693d-SA.json"

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]