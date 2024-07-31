FROM python:3.11-slim

RUN pip install mlflow==2.14.3 google-cloud-storage

ENV GCP_PROJECT=mlops-zoomcamp-423615

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]

