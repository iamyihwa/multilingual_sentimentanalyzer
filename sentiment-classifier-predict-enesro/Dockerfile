FROM ubuntu:latest
MAINTAINER Yihwa Kim "yihwa.kim.mail@gmail.com"
RUN apt-get clean  -y 
RUN apt-get update && apt-get install -y python3-pip python3-dev build-essential git-core
COPY . /app
WORKDIR /app
RUN cat ./model/pytorch_model.bin.zip.* > ./model/pytorch_model.bin.zip
RUN rm  ./model/pytorch_model.bin.zip.*
RUN unzip ./model/pytorch_model.bin.zip -d ./model/
RUN rm ./model/pytorch_model.bin.zip
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pip3 install -r requirements.txt
ENV FLASK_APP=sentiment-classifier-bert-predict.py
EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
