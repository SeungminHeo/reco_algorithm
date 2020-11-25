FROM python:3.7.8-slim-buster

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "ALS.py", "-re", "local", "--hours", "72"]