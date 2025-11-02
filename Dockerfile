FROM python:3.13.5

# Make uv available
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Create and install venv
COPY ./pyproject.toml /code/pyproject.toml
COPY ./uv.lock /code/uv.lock
RUN uv sync --locked --no-dev --compile-bytecode

COPY ./app /code/app

EXPOSE 80

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]