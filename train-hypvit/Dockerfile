FROM pytorch/torchserve:0.8.1-cpu
# FROM pytorch/torchserve:latest-cpu

ENV PYTHONUNBUFFERED True

WORKDIR /home/model-server/

# copy model artifacts, custom handler and other dependencies

COPY ./handler.py .
COPY ./utils.py ./bn_inception.py ./bn_inception-52deb4733.pth ./hypvit-config.yaml ./

# Switch to root user to make configuration changes
USER root

# Install required system packages and tools
RUN apt-get update && apt-get install -y libpq-dev build-essential git


# RUN export USE_CUDA=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip list

# Switch back to the model-server user
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081

# Package model artifacts and dependencies using TorchServe model archiver
RUN torch-model-archiver -f \
  --model-name=hypvit \
  --version=1.0 \
  --serialized-file=bn_inception-52deb4733.pth \
  --handler=handler.py \
  --extra-files=utils.py,bn_inception.py \
  --export-path=model-store \
  --config-file=hypvit-config.yaml

# Start TorchServe to serve prediction requests
CMD ["torchserve", "--start", "--models", "hypvit=hypvit.mar", "--model-store", "model-store"]
