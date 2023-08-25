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

def get_classes():
    with open('index_to_name.json', 'r') as json_file:
        classes = json.load(json_file)
    print(classes)
    return classes

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
        # self.model_file = "/home/desktop/Documents/torchserve-poc/train-detectron2/output/output_2/model_final.pth"
        self.model_file = "model_final.pth"
        self.config_file = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"  
        self.classes = None

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
            # initialize classes
            self.classes = get_classes()

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

        # images = []

        payloads = []

        # batch = [{'body': {'instances': [{'category': 'Womens Tshirt', 'image_url': 'http://....'}]}}]
        # batch is a list of requests
        for request in batch:
            # each item in the list is a dictionary with a single body key, get the body of the request
            request_body = request.get("body")
            request_instances = request_body.get("instances")

            for image_instance in request_instances:
                image_url = image_instance.get("image_url")
                category = image_instance.get("category")
                score = image_instance.get("score")

                print("category--->",category)

                if category == 'all':
                    categories = self.classes.values()
                else:
                    categories = category.split(',')

                is_categories_exist = all(value in self.classes.values() for value in categories)

                classes_string = ', '.join(str(value) for value in self.classes.values())

                if not is_categories_exist:
                    result = { "image_url":image_url,"error": f"Category not exist, use these classes ${classes_string}" }
                    payloads.append(result)
                    continue

            

                
                response = requests.get(image_url)
                response.raise_for_status()

                print(response)

                image_bytes = response.content



                input = io.BytesIO(image_bytes)
                img = cv2.imdecode(np.fromstring(input.read(), np.uint8), 1)
                # add the image to our list
                # images.append(img)

                result = {"image":img,"category":categories}

                payloads.append(result)


        print("pre-processing finished for a batch of {}".format(len(batch)))

        return payloads

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        # Do some inference call to engine here and return output
        print("inference started for a batch of {}".format(len(model_input)))

        output_payloads = []

        for payload in model_input:
            
            # generate bbox for image
            if "error" not in payload:
                image = payload['image']
                output = self.predictor(image)
                del payload['image']
                payload["output"] = output


            # run our predictions

            output_payloads.append(payload)

        print("inference finished for a batch of {}".format(len(model_input)))

        return output_payloads

    def postprocess(self, inference_output):

        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        start_time = time.time()
        
        print("post-processing started at {} for a batch of {}".format(start_time, len(inference_output)))
        
        responses = []

        for output_payload in inference_output:

            if "error" in output_payload:
                responses.append(output_payload)
                continue
            
            output = output_payload['output']
            categories = output_payload['category']

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

            # responses_json={'classes': classes, 'scores': scores, "boxes": boxes, "masks": masks }

            # Extract relevant information from the first box
            matching_index = None
            matching_category = None
            
            # get first matching class which is present in input categories
            for index,class_index in enumerate(classes):
                class_name = self.classes.get(str(class_index),None)

                # print(class_index,class_name,index)
                if class_name is None:
                    continue
                
                if class_name in categories:
                    matching_index = index
                    matching_category = class_name
                    break

            # if input category is not there return error
            if (matching_index is None) or (matching_category is None):
                output = {"error":"No matching index"}
                responses.append(output)
                continue
            

            # get bbox,score,and class_id
            first_box = boxes[matching_index]
            score = scores[matching_index]
            class_id = classes[matching_index]

            # Create the desired output dictionary
            output = {
                "box": first_box,
                "score": score,
                "class_id": class_id,
                "category": matching_category
            }

            # Print the desired output
            # print(output)

            responses.append(output)

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

        # output = []

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
        "predictions": output,
    }]


# #### THIS IS FOR RUNNING LOCALLY
# if __name__ == "__main__":

#     context = {
#         "system_properties": {
#             "batch_size": 1
#         }
#     }

#     if not _service.initialized:
#         _service.initialize(context)

#     # {
#     #     'image_url': "https://storage.googleapis.com/vera-hit/test.jpg".encode(),
#     #     'score': '0.5'.encode(),
#     #     'category': 'pants'.encode(),
#     #     'best_only': 'True'.encode(),
#     #     'mask': 'True'.encode(),
#     #     'color': "True".encode()
#     # }

    
#     data = {
                
#                 "instances": [
#                     {
#                         "best_only":"True",
#                         "mask":"True",
#                         "color":"True",
#                         "category": "Pants",
#                         "score": "0.5",
#                         "image_url": "https://storage.googleapis.com/vera-hit/test.jpg"
#                     },
#                 ]
                
#             }
            

#     print(data)

#     data = [{"body":data}]
    
#     output = _service.handle(data, context)
#     print("output------>",output)
#     # return json.dumps({"predictions": output})
