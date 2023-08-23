# custom service file

"""
ModelHandler defines a base model handler.
"""

# Some basic setup:
import detectron2
import os.path
import sys, io, json, time, random
import numpy as np
import cv2
import base64

import requests

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from os import path
from json import JSONEncoder
from detectron2 import model_zoo

classes = ['shirt',
           'top',
           'sweater',
           'cardigan',
           'jacket',
           'vest',
           'pants',
           'shorts',
           'skirt',
           'coat',
           'dress',
           'jumpsuit',
           'cape',
           'glasses',
           'hat',
           'headaccessory',
           'tie',
           'glove',
           'watch',
           'belt',
           'legwarmer',
           'stockings',
           'sock',
           'shoe',
           'bag',
           'scarf',
           'all'
           ]

valid_args = {
    'category': classes,
    'score': "score value for detectron ranging from 0 to 1",
    'best_only': ['True', 'False'],
    'mask': ['True', 'False'],
    'color': ['True', 'False']
}

valid_args_conditions = {
    'category': (lambda x: True if set(x.split(",")).issubset(set(classes)) else False),
    'score': (lambda x: True if 0 <= eval(x) <= 1 else False),
    'best_only': (lambda x: True if type(eval(x)) is bool else False),
    'mask': (lambda x: True if type(eval(x)) is bool else False),
    'color': (lambda x: True if type(eval(x)) is bool else False)
}

default_args = {
    'category': 'all',
    'score': '0.5',
    'best_only': 'True',
    'mask': 'False',
    'color': 'False'
}

def validate_arguments(data):
    valid_keys = valid_args_conditions.keys()
    for k, v in data.items():
        if k in ['image', 'url']:
            pass
        elif k in valid_keys:
            try:
                status = valid_args_conditions[k]
            except:
                status = False
            if not status:
                return {"error": {'Invalid argument passed.List of valid argument/value is': valid_args}}
        else:
            return {"error": {'Invalid argument passed.List of valid argument/value is': valid_args}}

    return True

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.predictor = None
        # self.model_file = "/home/desktop/Documents/torchserve-poc/train-detectron2/output/output_1/model_final.pth"
        self.model_file = "model_final.pth"
        self.config_file = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"  

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        print("initializing starting")

        print("File {} exists {}".format(self.model_file, str(path.exists(self.model_file))))
        print("File {} exists {}".format(self.config_file, str(path.exists(self.config_file))))

        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
            cfg.MODEL.WEIGHTS = self.model_file
            cfg.MODEL.DEVICE = 'cpu'

            # set the testing threshold for this model
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

            self.predictor = DefaultPredictor(cfg)

            print("predictor built on initialize")
        except AssertionError as error:
            # Output expected AssertionErrors.
            print(error)
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            print("Error: {}".format(e))

        self._context = context
        # self._batch_size = 1
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True
        print("initialized")

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))

        # Take the input data and pre-process it make it inference ready
        print("pre-processing started for a batch of {}".format(len(batch)))
        print("_batch_size == ",self._batch_size)

        images = []

        # batch = [{'body': {'instances': [{'categoryId': 'Womens Tshirt', 'image_url': 'http://....'}]}}]

    #     {
    #     'url': "https://storage.googleapis.com/vera-hit/test.jpg".encode(),
    #     'score': '0.5'.encode(),
    #     'category': 'pants'.encode(),
    #     'best_only': 'True'.encode(),
    #     'mask': 'True'.encode(),
    #     'color': "True".encode()
    # }
        # batch is a list of requests
        for request in batch:
            # each item in the list is a dictionary with a single body key, get the body of the request
            request_body = request.get("body")
            request_instances = request_body.get("instances")

            for image_instance in request_instances:
                image_url = image_instance.get("image_url")
                categoryId = image_instance.get("categoryId")
                
                response = requests.get(image_url)
                response.raise_for_status()

                image_bytes = response.content

                input = io.BytesIO(image_bytes)
                img = cv2.imdecode(np.fromstring(input.read(), np.uint8), 1)
                # add the image to our list
                images.append(img)


        print("pre-processing finished for a batch of {}".format(len(batch)))

        return images

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        # Do some inference call to engine here and return output
        print("inference started for a batch of {}".format(len(model_input)))

        outputs = []

        for image in model_input:
            # run our predictions
            output = self.predictor(image)

            outputs.append(output)

        print("inference finished for a batch of {}".format(len(model_input)))

        return outputs

    def postprocess(self, inference_output):

        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        start_time = time.time()
        
        print("post-processing started at {} for a batch of {}".format(start_time, len(inference_output)))
        
        responses = []

        for output in inference_output:

            # process predictions
            predictions = output["instances"].to("cpu")
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores.numpy() if predictions.has("scores") else None
            classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
            masks = (predictions.pred_masks > 0.5).squeeze().numpy() if predictions.has("pred_masks") else None

            if classes is not None:
                classes = classes.tolist()
            if scores is not None:
                scores = scores.tolist()
            if boxes is not None:
                boxes = [box.numpy().tolist() for box in boxes]
            if masks is not None:
                masks = masks.tolist()

            responses_json={'classes': classes, 'scores': scores, "boxes": boxes, "masks": masks }
			
            # print(responses_json)

            responses.append(responses_json)

        elapsed_time = time.time() - start_time
            
        print("post-processing finished for a batch of {} in {}".format(len(inference_output), elapsed_time))

        return responses

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print("handling started")
        print(data, "=====> data")
        # print(context.system_properties, " ====> context")
        # context.system_properties = {'model_dir': '/home/model-server/tmp/models/9d58942b2174498eae4e1f7e7a1e56fb', 'gpu_id': None, 'batch_size': 1, 'server_name': 'MMS', 'server_version': '0.8.1', 'limit_max_image_pixels': True}
        
        
        # print("handle data ----> ")
        # print("handle data ----> ", data) asia-south1-docker.pkg.dev/vera-dev-392610/detectron-containers
        # process the data through our inference pipeline
        model_input = self.preprocess(data)
        

        # print("model_input ----> ")
        # print("model_input ----> ", model_input)
        model_out = self.inference(model_input)
        
        # print("output data ----> ")
        # print("output data ----> ", model_out)
        output = self.postprocess(model_out)

        # print("handling finished")
        print("handling finished", output)

        return output  

_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        print("service not initialized")
        _service.initialize(context)

    if data is None:
        return None
    
    output = _service.handle(data, context)

    return [{
        "predictions": output
    }]


##### THIS IS FOR RUNNING LOCALLY
# if __name__ == "__main__":

#     context = {
#         "system_properties": {
#             "batch_size": 1
#         }
#     }

#     if not _service.initialized:
#         _service.initialize(context)

#     data = [
#                 {
#                     "body": {
#                     "instances": [
                        # {
                        # "categoryId": "Womens Tshirt",
                        # "image_url": "https://hips.hearstapps.com/hmg-prod/images/edc110122grenney-003-1666121032.jpg"
                        # },
                        # {
                        # "categoryId": "Womens Tshirt",
                        # "image_url": "https://hips.hearstapps.com/hmg-prod/images/edc110122grenney-003-1666121032.jpg"
                        # },
                        # {
                        # "categoryId": "Womens Tshirt",
                        # "image_url": "https://hips.hearstapps.com/hmg-prod/images/edc110122grenney-003-1666121032.jpg"
                        # }
#                     ]
#                     }
#                 }
#             ]
    
#     output = _service.handle(data, context)
#     print(json.dumps({"predictions": output}))
#     return json.dumps({"predictions": output})
