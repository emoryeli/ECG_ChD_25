FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install dependencies using apt install, etc.

## install python tools
RUN pip install -r requirements.txt
