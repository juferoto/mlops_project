FROM tiangolo/uvicorn-gunicorn:python3.11-slim

WORKDIR /app
ADD . /app

RUN python -m venv venv

RUN /bin/bash -c "source venv/bin/activate"

RUN pip install -r requirements.txt