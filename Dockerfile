FROM python:3.8-slim

COPY .  /app
WORKDIR /app

ADD requirements.txt ./requirements.txt
RUN apt-get -y update
RUN apt-get -y install python3-opencv
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["streamlit","run"]
CMD ["first_app.py"]
