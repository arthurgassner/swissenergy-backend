FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./model_server /code/model_server

CMD ["fastapi", "run", "model_server/main.py", "--port", "81"]