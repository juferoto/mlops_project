FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONBUFFERED=1

WORKDIR /app
ADD . /app

RUN apt-get update
RUN python -m venv venv

RUN /bin/bash -c "source venv/bin/activate"

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "create_service.py"]
