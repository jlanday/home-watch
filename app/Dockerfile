# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=America/Los_Angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
    git \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libpng-dev \
    ffmpeg \
    python3-opencv 

WORKDIR /app
RUN cd /app

ADD *.py /app/
ADD *.json /app/
ADD *.txt /app/

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt

CMD tail -f /dev/null
