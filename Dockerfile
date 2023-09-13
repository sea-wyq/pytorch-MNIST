FROM nvidia/cuda

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install requirements.txt

COPY . .

CMD ["python3", "train.py"]