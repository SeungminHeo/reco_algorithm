FROM python:3.7.8-slim-buster

RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y --reinstall build-essential

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "ALS_FB.py"]