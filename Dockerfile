FROM python:3.8.8

ENV POETRY_VIRTUALENVS_CREATE=0
ENV POETRY_VERSION=1.1.5

RUN python -m pip install poetry==${POETRY_VERSION}

WORKDIR /code
COPY poetry.lock pyproject.toml /code/

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

RUN apt-get update \
 && apt-get install -y libpq-dev \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p data
COPY ./data/dataset.csv ./data/
COPY ./src ./src
COPY ./build_and_run_project.py ./

CMD ["poetry", "run", "python", "build_and_run_project.py" ]