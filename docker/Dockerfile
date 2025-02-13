FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Make sure that lrrnn can be imported
ENV PYTHONPATH=/workspace

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip --no-cache-dir install -r ./requirements.txt && \
    rm requirements.txt

# Install Git for development
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
