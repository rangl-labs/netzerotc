# Dockerfile for the rangl environment server
FROM python:3.9-slim-buster

WORKDIR /service
COPY rangl/requirements.txt .
RUN pip install -r requirements.txt

COPY . /service
RUN pip install .
RUN pip list

CMD ["python", "rangl/server.py"]