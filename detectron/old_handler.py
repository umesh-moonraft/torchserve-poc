######################################
# Author [Gaurav Singh , Trinanjan Saha]
# 18/09/2020
# This code is used for serving detectron_batching model
######################################

######################################
# Imports
######################################
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import json
import io
import logging
import numpy as np
import os
import torch
from PIL import Image
import pycocotools.mask as rle_mask
import requests
import cv2
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import MiniBatchKMeans
import time
######################################
# SETTINGS AND GLOBAL VARIABLES
######################################
logger = logging.getLogger(__name__)

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

######################################
# HELPER FUNCTIONS
######################################


def get_image(data):
    input_image = data.get("image", None)
    if input_image == None:
        input_image_url = data.get("url", None).decode('utf-8')
        if input_image_url == None:
            return {"error": "Image / URL missing"}
        else:
            try:
                response = requests.get(input_image_url, headers={
                                        'User-Agent': 'My User Agent 1.0'})
            except Exception as e:
                print(e)
                return {"error": "Unable to download image from URL"}
            else:
                try:
                    pil_image = Image.open(io.BytesIO(
                        response.content)).convert('RGB')
                except Exception as e:
                    print(e)
                    return {"error": "Inavalid image"}
    else:
        try:
            pil_image = Image.open(io.BytesIO(input_image)).convert('RGB')
        except Exception as e:
            print(e)
            return {"error": "Inavalid image"}
    return {'pil_image': pil_image}


def validate_arguments(data):
    input_args = {}
    valid_keys = valid_args_conditions.keys()
    for k, v in data.items():
        if k in ['image', 'url']:
            pass
        elif k in valid_keys:
            try:
                status = valid_args_conditions[k](v.decode('utf-8'))
            except:
                status = False
            if status:
                input_args[k] = v.decode('utf-8')
            else:
                return {"error": {'Invalid argument passed.List of valid argument/value is': valid_args}}
        else:
            return {"error": {'Invalid argument passed.List of valid argument/value is': valid_args}}
    return input_args


def parse_request(data):
    payloads = []
    for d in data:
        payload = {}
        payload = get_image(d)
        if 'pil_image' in payload:
            args = validate_arguments(d)
            if 'error' in args:
                payloads.append(args)
            else:
                args = {**default_args, **args}
                payload.update(args)
                payloads.append(payload)
        else:
            payloads.append(payload)
    return payloads


def get_color(image, mask, sz=64):
    img = np.array(image.resize((sz, sz), Image.NEAREST))
    mask = np.array(mask, dtype=np.int)
    mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
    pixdata = np.compress(mask.flatten(), img.reshape(-1, 3), axis=0)
    labs = rgb2lab([pixdata]).squeeze()

    if len(labs.shape) == 2:
        sample_size = labs.shape[0]
    elif len(labs.shape) == 1:
        sample_size = 1
    else:
        sample_size = 0

    if sample_size > 3:
        # seeding = kmc2.kmc2(labs, 3)
        kmeans = MiniBatchKMeans(n_clusters=3)
        kmeans.fit(np.squeeze(labs))
        LABELS = kmeans.labels_
        colors = kmeans.cluster_centers_
        colors = colors[np.argsort(
            np.unique(LABELS, return_counts=True)[1])[::-1]]
        return (colors.tolist(), (lab2rgb([colors]).squeeze()*255).tolist())
    if sample_size == 0:
        return ([], [])
    else:
        return (labs.tolist(), (lab2rgb([labs]).squeeze()*255).tolist())

######################################
# DETECTRON MAIN CLASS
######################################


class Detectron2(object):

    def __init__(self):
        self.model = None
        self.initialized = False
        self.device = False

    def initialize(self, model_dir, gpu_id):
        self.device = torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        model_pt_path = os.path.join(model_dir, "model_final.pth")
        model_conf_path = os.path.join(
            model_dir, "mask_rcnn_X_101_32x8d_FPN_3x.yaml")\

        cfg = get_cfg()
        cfg.merge_from_file(model_conf_path)
        cfg.MODEL.WEIGHTS = model_pt_path
        self.model = build_model(cfg)
        DetectionCheckpointer(self.model).load(model_pt_path)
        self.model.train(False)

        logger.debug(
            'Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, payloads):
        local_payloads = []
        for payload in payloads:
            if 'error' in payload:
                local_payloads.append(payload)
            else:
                local_payload = payload.copy()
                local_payload.update({'cv_image': (
                    np.array(payload['pil_image'].convert('RGB'))[:, :, ::-1])})  # input is BGR
                local_payloads.append(local_payload)
        return local_payloads

    def inference(self, payloads):
        local_payloads = []
        for payload in payloads:
            if 'cv_image' in payload:
                temp_payoad = np.transpose(
                    payload['cv_image'], axes=(-1, 0, 1))
                local_payloads.append(
                    {"image": torch.tensor(temp_payoad.copy())})

        if len(local_payloads):
            outputs = self.model(local_payloads)
        else:
            outputs = []

        output_payloads = []
        for index, payload in enumerate(payloads):
            if 'error' in payload:
                output_payloads.append(payload)
            else:
                payload.update(outputs[index])
                output_payloads.append(payload)

        return output_payloads

    def postprocess(self, payloads):
        final_data = []
        for payload in payloads:
            if 'error' in payload:
                final_data.append(payload)
            else:
                instances = payload['instances']
                pred_boxes = instances.pred_boxes.tensor.detach().cpu().tolist()
                scores = instances.scores.detach().cpu().tolist()
                pred_classes = instances.pred_classes.detach().cpu().tolist()
                pred_mask = instances.pred_masks.detach().cpu().tolist()
                if payload['category'] == 'all':
                    payload['classes'] = classes
                else:
                    payload['classes'] = payload['category'].split(",")

                results = [pred_boxes, scores, pred_classes, pred_mask]

                # if(eval(payload['color'])):
                #     color=[  dict(zip( ("LAB","RGB"),get_color(payload['pil_image'],mask)))  for mask in instances.pred_masks.detach().cpu().tolist()]
                #     results.append(color)

                # if eval(payload['mask']):
                #     lre_encoded_masks= [ rle_mask.encode(np.asfortranarray(mask,dtype=np.uint8)) for mask in instances.pred_masks.detach().cpu().tolist() ]
                #     serializable_masks=[]
                #     for mask in lre_encoded_masks:
                #         tmp={}
                #         tmp['size']=mask['size']
                #         tmp['counts'] = mask['counts'].decode("utf-8")
                #         serializable_masks.append(tmp)
                #     results.append(serializable_masks)

                results = zip(*results)
                items = []
                keys = []
                for index, result in enumerate(results):
                    result[0][0] = result[0][0]/payload['cv_image'].shape[1]
                    result[0][1] = result[0][1]/payload['cv_image'].shape[0]
                    result[0][2] = result[0][2]/payload['cv_image'].shape[1]
                    result[0][3] = result[0][3]/payload['cv_image'].shape[0]
                    # result[0]=result[0].tolist()
                    class_name = classes[result[2]]
                    if class_name not in payload['classes']:
                        continue
                    if result[1] <= eval(payload['score']):
                        continue
                    final_result = {}
                    final_result['box'] = result[0]
                    final_result['score'] = result[1]
                    final_result['class_id'] = result[2]
                    final_result['category'] = class_name

                    if (eval(payload['color'])):
                        color = dict(zip(("LAB", "RGB"), get_color(
                            payload['pil_image'], result[3])))
                        final_result['color'] = color

                    if eval(payload['mask']):
                        lre_encoded_mask = rle_mask.encode(
                            np.asfortranarray(result[3], dtype=np.uint8))
                        tmp = {}
                        tmp['size'] = lre_encoded_mask['size']
                        tmp['counts'] = lre_encoded_mask['counts'].decode(
                            "utf-8")
                        final_result['mask'] = tmp

                    if class_name not in keys:
                        items.append(final_result)
                        keys.append(class_name)
                    else:
                        if eval(payload['best_only']):
                            existing_entry = list(
                                filter(lambda x: class_name == x['category'], items))[0]
                            if result[1] > existing_entry['score']:
                                items.remove(existing_entry)

                                items.append(final_result)
                        else:
                            items.append(final_result)
                final_data.append(items)

        # for cat in payload['category'].split(","):
        #     if cat not in keys and cat !="all":
        #         items.append({cat:"NA"})
        return final_data


_service = Detectron2()

######################################
# PYTORCH SERVE FUNCTION
######################################


def handle(data, context):

    if not _service.initialized:
        properties = context.system_properties
        gpu_id = properties.get("gpu_id")
        model_dir = properties.get("model_dir")
        _service.initialize(model_dir=model_dir, gpu_id=gpu_id)
        print(model_dir, "===========>Initialized")
    if data is None:
        return [{"error": "No input given."}]
    else:
        tic = time.time()
        payloads = parse_request(data)
        payloads = _service.preprocess(payloads)
        payloads = _service.inference(payloads)
        payloads = _service.postprocess(payloads)
        toc = time.time()
        print("time taken by detectron for ", len(data), " items is ", toc-tic)
        return payloads


######################################
# MAIN FUNCTION
######################################
if __name__ == "__main__":

    if not _service.initialized:
        _service.initialize(model_dir=".", gpu_id=0)
    data = [{
        'url': "https://storage.googleapis.com/vera-hit/test.jpg".encode(),
        'score': '0.5'.encode(),
        'category': 'pants'.encode(),
        'best_only': 'True'.encode(),
        'mask': 'True'.encode(),
        'color': "True".encode()
    }]*4
    payload = parse_request(data)
    if "error" in payload:
        print([{"error": payload['error']}])
    else:
        payload = _service.preprocess(payload)
        payload = _service.inference(payload)
        payload = _service.postprocess(payload)
        print(json.dumps(payload, indent=4))
