FROM tiangolo/uvicorn-gunicorn:python3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONBUFFERED=1

WORKDIR /app
ADD . /app

RUN python -m venv venv

RUN /bin/bash -c "source venv/bin/activate"

RUN pip install -r requirements.txt
