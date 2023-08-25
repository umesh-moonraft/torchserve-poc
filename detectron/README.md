torch-model-archiver --model-name detectron --version 1.0 --serialized-file /home/desktop/Documents/torchserve-poc/train-detectron2/output/output_1/model_final.pth --handler object_detector --extra-files index_to_name.json --export-path /model_store --config-file config.yaml

torchserve --start --model-store model_store --models detectron=detectron.mar --ncs

Request image_url, category: ['Womens Tshirt']
Response
{
image_url: "",
category: {
bbox: []
}
}

Request image_url, category: ['Womens Tshirt', 'Laptops']
Response
{
image_url: "",
category('womens Tshirt'): {
bbox: []
},
category: {
bbox: []
}

}

docker tag detectron:latest asia-south1-docker.pkg.dev/vera-dev-392610/detectron-containers/detectron
docker push asia-south1-docker.pkg.dev/vera-dev-392610/detectron-containers/detectron

docker tag vera-model:latest asia-south1-docker.pkg.dev/vera-dev-392610/vera-models-container/vera-model
docker push asia-south1-docker.pkg.dev/vera-dev-392610/vera-models-container/vera-model
