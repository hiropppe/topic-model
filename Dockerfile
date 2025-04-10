FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /root

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config locales tzdata \
    python3 python3-dev python3-pip python3-wheel python3-venv pipx \
    vim git curl jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pipx ensurepath

RUN locale-gen ja_JP.UTF-8

ENV LANG ja_JP.UTF-8
ENV TZ Asia/Tokyo
ENV PATH="${PATH}:/root/.local/bin"

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pipx install poetry

COPY . /text-model

WORKDIR /text-model
RUN poetry install

#CMD ["poetry", "run", "jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=''", "--no-browser"]
