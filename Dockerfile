# FROM pytorch/torchserve:0.8.1-cpu
FROM pytorch/torchserve:latest-cpu

ENV PYTHONUNBUFFERED True

WORKDIR /home/model-server/


# Proxy Anchor Directory (Copy files) /home/model-server/proxy-anchor
COPY ./proxy-anchor/handler.py ./proxy-anchor
COPY ./proxy-anchor/utils.py ./proxy-anchor/bn_inception.py ./proxy-anchor/output/bn_inception-52deb4733.pth ./proxy-anchor/hypvit-config.yaml ./proxy-anchor


# Detectron Directory (Copy files) /home/model-server/detectron
COPY ./detectron/handler.py ./detectron
COPY ./detectron/index_to_name.json ./detectron
COPY ./detectron/output/output_2/model_final.pth ./detectron
COPY ./detectron/detectron-config.yaml ./detectron

# Switch to root user to make configuration changes
USER root

# Install required system packages and tools
RUN apt-get update && apt-get install -y libpq-dev build-essential git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Switch back to the model-server user
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081


# Package model artifacts and dependencies using TorchServe model archiver [Proxy Anchor]
RUN torch-model-archiver -f \
  --model-name=proxy-anchor \
  --version=1.0 \
  --serialized-file=/home/model-server/proxy-anchor/bn_inception-52deb4733.pth \
  --handler=/home/model-server/proxy-anchor/handler.py \
  --extra-files  "/home/model-server/proxy-anchor/utils.py,/home/model-server/proxy-anchor/bn_inception.py" \
  --export-path=/home/model-server/model-store \
  --config-file=/home/model-server/proxy-anchor/hypvit-config.yaml

# create model archive file packaging model artifacts and dependencies [Detectron]
RUN torch-model-archiver -f \
  --model-name=detectron \
  --version=1.0 \
  --serialized-file=/home/model-server/detectron/model_final.pth \
  --handler=/home/model-server/detectron/handler.py \
  --extra-files=/home/model-server/detectron/index_to_name.json \
  --export-path=/home/model-server/model-store \
  --config-file=/home/model-server/detectron/detectron-config.yaml

# Start TorchServe to serve prediction requests
CMD ["torchserve", "--start", "--models", "proxy-anchor=proxy-anchor.mar" ,"detectron=detectron.mar" , "--model-store", "/home/model-server/model-store"]
