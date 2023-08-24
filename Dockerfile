# FROM pytorch/torchserve:0.8.1-cpu
FROM pytorch/torchserve:latest-cpu

ENV PYTHONUNBUFFERED True

WORKDIR /home/model-server/

# Creating both the model directory
RUN mkdir /home/model-server/proxy-anchor
RUN mkdir /home/model-server/detectron

# Proxy Anchor Directory (Copy files) /home/model-server/proxy-anchor
COPY ./proxy-anchor/handler.py /home/model-server/proxy-anchor
COPY ./proxy-anchor/utils.py /home/model-server/proxy-anchor
COPY ./proxy-anchor/bn_inception.py /home/model-server/proxy-anchor
COPY ./proxy-anchor/output/bn_inception-52deb4733.pth /home/model-server/proxy-anchor
COPY ./proxy-anchor/proxy-anchor-config.yaml /home/model-server/proxy-anchor


# Detectron Directory (Copy files) /home/model-server/detectron
COPY ./detectron/handler.py /home/model-server/detectron
COPY ./detectron/index_to_name.json /home/model-server/detectron
COPY ./detectron/output/output_2/model_final.pth /home/model-server/detectron
COPY ./detectron/detectron-config.yaml /home/model-server/detectron

# Switch to root user to make configuration changes
USER root

# Install required system packages and tools
RUN apt-get update && apt-get install -y libpq-dev build-essential git

# Installing requirements.txt
COPY ./docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Switch back to the model-server user
USER model-server

# Navigating to Proxy Anchor directory
WORKDIR /home/model-server/proxy-anchor

# Package model artifacts and dependencies using TorchServe model archiver [Proxy Anchor]
RUN torch-model-archiver -f \
  --model-name=proxy-anchor \
  --version=1.0 \
  --serialized-file=./bn_inception-52deb4733.pth \
  --handler=./handler.py \
  --extra-files  "./utils.py,./bn_inception.py" \
  --export-path=/home/model-server/model-store \
  --config-file=./proxy-anchor-config.yaml

# Navigating to detectron directory
WORKDIR /home/model-server/detectron

# create model archive file packaging model artifacts and dependencies [Detectron]
RUN torch-model-archiver -f \
  --model-name=detectron \
  --version=1.0 \
  --serialized-file=./model_final.pth \
  --handler=./handler.py \
  --extra-files=./index_to_name.json \
  --export-path=/home/model-server/model-store \
  --config-file=./detectron-config.yaml



# Setting Root Directory
WORKDIR /home/model-server/

# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081


# Start TorchServe to serve prediction requests
CMD ["torchserve", "--start","--ts-config", "/home/model-server/config.properties", "--models", "proxy-anchor=proxy-anchor.mar detectron=detectron.mar" , "--model-store", "/home/model-server/model-store"]