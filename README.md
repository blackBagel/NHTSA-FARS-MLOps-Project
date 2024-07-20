# NHTSA-FARS-MLOps-Project


For now i'll use mlflow locally and it'll have a gcp bucket as an artifact storage:

```bash
mlflow server \
    --default-artifact-root gs://mlops_zoomcamp-mlflow-artifacts/artifacts \
    --backend-store-uri sqlite:///${PWD}/data/mlflow/mlflow.db \
    --host 0.0.0.0 \
    --port 5000
```