FROM ubuntu:18.04

USER root

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive \
  apt-get -y -q install \
  build-essential git \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg2 \
  software-properties-common \
  python3 python3-dev python3-pip \
  openjdk-8-jre

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH=$PYTHONPATH:/opt/wine_prediction
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /opt/wine_prediction

COPY . .
RUN pip3 install -r requirements.txt
