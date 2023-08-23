# FROM pytorch/torchserve:0.8.1-gpu
FROM pytorch/torchserve:latest-cpu

ENV PYTHONUNBUFFERED True

# copy model artifacts, custom handler and other dependencies
COPY ./handler.py /home/model-server/
COPY ./index_to_name.json /home/model-server/
COPY ./output/output_1/model_final.pth /home/model-server/
COPY ./detectron-config.yaml /home/model-server/
# COPY ./config.properties /home/model-server/
# COPY ./model/$APP_NAME/ /home/model-server/

# create torchserve configuration file
USER root
# RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
# RUN printf "\ninference_address=http://0.0.0.0:8080" >> /home/model-server/config.properties
# RUN printf "\nmanagement_address=http://0.0.0.0:8081" >> /home/model-server/config.properties


# Install core dependencies.
RUN apt-get update && apt-get install -y libpq-dev build-essential && apt-get install -y git

# RUN export USE_CUDA=1
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

USER model-server

# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=detectron \
  --version=1.0 \
  --serialized-file=/home/model-server/model_final.pth \
  --handler=/home/model-server/handler.py \
  --extra-files=/home/model-server/index_to_name.json \
  --export-path=/home/model-server/model-store \
  --config-file=/home/model-server/detectron-config.yaml

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
    #  "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "detectron=detectron.mar", \
     "--model-store", \
     "/home/model-server/model-store"]
