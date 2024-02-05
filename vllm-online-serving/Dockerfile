FROM quay.io/vessl-ai/python:3.10-r2

ENV PROMETHEUS_VERSION=2.49.1

WORKDIR /app

# Download and unzip prometheus binary to /app/prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
RUN tar -xvf prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
RUN mv prometheus-$PROMETHEUS_VERSION.linux-amd64 ./prometheus
RUN rm prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz

# Copy prometheus.yml and grafana.json
COPY monitoring/prometheus.yml /app/prometheus/prometheus.yml

# # Install dependencies
# COPY requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt

# # Entrypoint
# ENTRYPOINT ["python", "-m", "api.py"]
