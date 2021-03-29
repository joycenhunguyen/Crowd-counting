FROM python:3.8-slim

RUN apt-get -y update
RUN apt-get -y install python3-opencv

WORKDIR /srv
ADD ./requirements.txt /srv/requirements.txt
RUN pip install -r requirements.txt

ADD . /srv

EXPOSE 8080

ENTRYPOINT ["streamlit","run"]
CMD ["first_app.py"]