FROM python:3.11.9-slim

RUN pip install mlflow==2.14.2 google-cloud-storage==2.18.1

ENV GOOGLE_APPLICATION_CREDENTIALS="/run/secrets/mlops-zoomcamp-mlflow-SA-423615-aa494484f08e.json"

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--default-artifact-root", "gs://mlops_zoomcamp-mlflow-artifacts/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]