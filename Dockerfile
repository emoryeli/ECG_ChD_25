FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install dependencies using apt install, etc.

## install python tools
RUN pip install -r requirements.txt
# After running requirements.txt, install torch with CUDA.
# RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
