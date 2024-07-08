FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.10 \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /code
COPY ./requirements.txt /code/

RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv
RUN python3.10 -m uv pip install --no-cache-dir -r requirements.txt

COPY ./service.py /code/
COPY ./bentofile.yaml /code/

EXPOSE 3000

CMD [ "bentoml", "serve", "." ]
