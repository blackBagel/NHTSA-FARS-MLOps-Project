FROM prefecthq/prefect:2.20-python3.11

EXPOSE 4200

RUN prefect config set PREFECT_API_DATABASE_CONNECTION_URL='sqlite+aiosqlite:////database/prefect.db'
RUN prefect config set PREFECT_SERVER_API_HOST='0.0.0.0'

CMD [ "prefect", "server", "start"]