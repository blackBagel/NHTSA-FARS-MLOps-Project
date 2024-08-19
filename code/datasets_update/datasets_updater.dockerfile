FROM python:3.11.9-slim

RUN pip install -U pip
RUN pip install -U pandas prefect

WORKDIR /app

ENV FILE_NAME = 'person.csv'
ENV DATASETS_DIR_RELATIVE_PATH = '/datasets'
ENV PREFECT_API_URL="http://prefect:4200/api"

COPY [ "code/datasets_update", "./" ]

ENTRYPOINT [ "python", "update_train_val_test_data.py" ]