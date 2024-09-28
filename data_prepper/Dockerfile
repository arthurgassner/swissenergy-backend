FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./data_prepper /code/data_prepper

CMD ["fastapi", "run", "data_prepper/main.py", "--port", "80"]