
######################################
# Author [Gaurav Singh , Trinanjan Saha]
# 24/08/2020
# This code is used for serving proxy anchor model
# TODO include batching
######################################

######################################
# Imports
######################################

# proxy anchor model specific imports
import torch
import utils
from bn_inception import *
from tqdm.auto import tqdm

# other imports
import logging
import json
import io
import logging
import numpy as np
import os
from PIL import Image
import requests
import base64
import time
import json
import gc



######################################
# SETTINGS
######################################

# Initialize logger and Arguments
logger = logging.getLogger(__name__)

# Valid arguments and their condition
valid_args = {
    'bbox': 'bbox should be list of normalized/absolute values with the format [x1,y1,x2,y2], example [.1,.2,.7,.8]/[10,20,100,120]'
}

valid_args_conditions = {
    'bbox': (lambda x: True if type(eval(x)) is list else False)
}


# no default args. user has the option to pass either url/image_path and bbox or only the url/image_path
default_args = {}

######################################
# HELPER FUNCTIONS
######################################


def get_image(data):
    """
    1. input argument data=[{
    'url':"https://www.thetrendspotter.net/wp-content/uploads/2019/09/London-Fashion-Week-SS-2020-Street-Style-34.jpg".encode(),
    'bbox':[.1,.2,.7,.8]
        }]
    2. output PILimage/corrupted image flag
    3. get_image function takes the payload checks wheather the image url is given or the image path is given
    4. Accordingly it lodes the image and returns the image
    5. It also checks if the image is corrupted
    """
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
            # modifed[Ammu]
            image = Image.open(input_image)
            pil_image = image.convert('RGB')
            # pil_image = Image.open(io.BytesIO(input_image)).convert('RGB')
        except Exception as e:
            print(e)
            return {"error": "Inavalid image"}
    return {'pil_image': pil_image}


def validate_arguments(data, payload):
    """
    1. input argument 

    data=[{'url':"https://www.thetrendspotter.net/wp-content/uploads/2019/09/London-Fashion-Week-SS-2020-Street-Style-34.jpg".encode(),
    'bbox':[.1,.2,.7,.8]}]
    payload {'pil_image': <PIL.Image.Image image mode=RGB size=600x900 at 0x7FBA159EDE20>

    2. output validated arguments, in this case bbox sanity check

    3. Error checking for case

        a. checks if the bbox is list
        b. checks if the bbox length is exactly 4
        c. checks if the bbox values are normalized, if normalized convert them to absolute
        d. checks if the bbox values are in logical range

        TODO 
        e. use the bbox_sanity_check flag to return differnet error messages,currently only invalid args error statement is being returned
           It will be better to notify the user that bbox range is logically wrong. For example if the last y2  value is more than the image 
           height etc. 
    """
    input_args = {}
    valid_keys = valid_args_conditions.keys()

    for k, v in data.items():
        if k in ['image', 'url']:
            pass
        elif k in valid_keys:

            try:
                # (lambda x: True if type(eval(x)) is list else False)
                status = valid_args_conditions[k](v)

                # check if the image bbox values are normalized
                bbox = eval(v)
                # length bbox check
                bbox_sanity_check = True if len(bbox) == 4 else False

                if bbox_sanity_check:
                    # normalized check
                    for cordinate in bbox:
                        if cordinate > 1:
                            normalized = False
                        else:
                            normalized = True

                    img_width, img_height = payload['pil_image'].size

                    # If normalized then remove normalization
                    if normalized:
                        # check the bbox range
                        if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] > 1 or bbox[3] > 1):
                            bbox_sanity_check = False
                        else:
                            bbox_processed = [
                                bbox[0]*img_width, bbox[1]*img_height, bbox[2]*img_width, bbox[3]*img_height]
                    # else use the same bbox values
                    else:
                        # check the bbox range
                        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > img_width or bbox[3] > img_height:
                            bbox_sanity_check = False
                        else:
                            bbox_processed = bbox

                    # cropped image
                    cropped_image = payload['pil_image'].crop((bbox_processed))
                    # check if the cropped image is valid or not
                    # this also ensures if the crop is logically wrong or outside the image
                    if cropped_image.tobytes():
                        bbox_sanity_check = True

            except:
                status = False
                bbox_sanity_check = False  # TODO use this flag to return logical bbox value error
            if status and bbox_sanity_check:
                input_args[k] = v
                # The following new key_values are added to the payload
                # This crop pil image is directly being used inside the ProxyAnchor class preprocess function
                input_args['pil_image_cropped'] = cropped_image
                input_args['pil_image_cropped_bbox'] = bbox_processed
            else:
                return {"error": {'Invalid argument passed.List of valid argument/value is': valid_args}}
        else:
            return {"error": {'Invalid argument passed.List of valid argument/value is': valid_args}}
    return input_args


def parse_request(data):
    """
    1. input argument data=[{
    'url':"https://www.thetrendspotter.net/wp-content/uploads/2019/09/London-Fashion-Week-SS-2020-Street-Style-34.jpg".encode(),
    'bbox':[.1,.2,.7,.8]
        }]
    2. output --> after validation the final processed payload for proxy_anchor model 
    """
    payloads = []
    for d in data:
        payload = {}
        payload = get_image(d)
        if 'pil_image' in payload:
            args = validate_arguments(d, payload)
            if 'error' in args:
                payload['image'] = d.image
                payloads.append(args)
            else:
                args = {**default_args, **args}
                payload.update(args)
                payload['image'] = d['image']
                payloads.append(payload)
        else:
            payloads.append(payload)
    return payloads


class ProxyAnchor(object):

    """
        # TODO add comments for this section
        # preprocess function checks for bbox key present or not
    """

    def __init__(self):
        self.model = None
        self.initialized = False
        self.device = False
        self.transform = utils.make_transform(is_train=False, is_inception=1)

    def initialize(self, model_dir, gpu_id):
        # modifed[Ammu]
        # self.device = torch.device(
        #     "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # modifed[Ammu]
        # model_pt_path = os.path.join(model_dir, "Inshop_bn_inception_best.pth")
        model_pt_path = os.path.join(model_dir, "bn_inception-52deb4733.pth")
        self.model = bn_inception(
            embedding_size=512, pretrained=True, is_norm=1, bn_freeze=1)

        # checkpoint = torch.load(model_pt_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        if (self.device == "cpu"):
            print('CPU')
            self.model.load_state_dict(torch.load(
                model_pt_path, map_location="cpu"), strict=False)
        else:
            print('GPU')
            checkpoint = torch.load(model_pt_path)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False)

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
                # check if bbox is present and crop the image
                if 'bbox' in local_payload.keys():
                    input_img = payload['pil_image_cropped']
                # otherwise pass the same image
                else:
                    input_img = payload['pil_image']

                # print(input_img.size, type(input_img), len(input_img.mode), input_img.mode)

                local_payload.update({'pil_processed_image': (
                    self.transform(input_img).unsqueeze(0))})
                local_payloads.append(local_payload)
        return local_payloads

    def inference(self, payloads):
        local_payloads = []

        for payload in payloads:
            if 'pil_processed_image' in payload:
                local_payloads.append(payload['pil_processed_image'])
        if len(local_payloads):
            outputs = self.model(torch.cat(local_payloads))
            outputs = list(torch.split(outputs, 1))
        else:
            outputs = []
        output_payloads = []
        for payload in payloads:
            if 'error' in payload:
                output_payloads.append(payload)
            else:
                output_payloads.append(
                    {'instances': outputs.pop(0), 'image': payload['image']})
        return output_payloads

    def postprocess(self, payloads):
        final_data = []
        for payload in payloads:
            if 'error' in payload:
                final_data.append(payload)
            else:
                instances = payload['instances']
                result = instances.detach().cpu().numpy().squeeze()
                data = {"embedding": {"data": result.tolist(
                ), "version": 5, 'image': payload['image']}}
                final_data.append(data)
        return final_data


_service = ProxyAnchor()


def handle(data, context):
    """
        a. TODO comment should be added by Gaurav
        a. Following the same code from the torchserve repo written by Gaurav
    """
    if not _service.initialized:
        properties = context.system_properties
        gpu_id = properties.get("gpu_id")
        model_dir = properties.get("model_dir")
        _service.initialize(model_dir=model_dir, gpu_id=gpu_id)

    if data is None:
        return [{"error": "No input given."}]
    else:
        payloads = parse_request(data)
        payloads = _service.preprocess(payloads)
        payloads = _service.inference(payloads)
        payloads = _service.postprocess(payloads)
        return payloads


if __name__ == "__main__":
    """
        1. Main function 
        2. Assembles the entire code to get embedding for one image
    """
    gc.set_threshold(0)
    if not _service.initialized:
        # _service.initialize(
        #     model_dir='/home/guest/Documents/piktor-vera-pipeline-torchserve-5d1b7e93cf9c/data/proxy_anchor', gpu_id=0)
        _service.initialize(
            model_dir='/home/user/Documents/Vera/piktor-vera-pipeline-torchserve-hypvitembedding/data/proxy_anchor/', gpu_id=0)

    # data = [{
    #     "url": "https://i.pinimg.com/736x/e5/15/ae/e515ae565b823c87b684b6904d9478a3.jpg".encode(),
    #     'bbox': '[0.1731438191731771, 0.6382481384277344, 0.7134397379557291, 0.9916904703776042]'
    # }]
    # modifed[Ammu]
    file_path = "proxy_anchor_input_data.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    # data = [{
    #     "image": '/Users/apple/Documents/Vera/vera-Dataset-for-Training/Curtain/0a1e229710d82db4.jpg',
    #     "url": "https://i.pinimg.com/736x/e5/15/ae/e515ae565b823c87b684b6904d9478a3.jpg".encode(),
    #     'bbox': '[0.1731438191731771, 0.6382481384277344, 0.7134397379557291, 0.9916904703776042]'
    # }]
    # test payload
    # data = [{
    #     'image' : '/media/pintu/BACKUP/Trinanjan/current_project/virtual_try_on/Graphonomy/img/messi.jpg',
    #     'bbox':'[.1,.2,.7,.8]'
    # }]
    # data=[{
    # 'url':"https://www.thetrendspotter.net/wp-content/uploads/2019/09/London-Fashion-Week-SS-2020-Street-Style-34.jpg".encode(),
    # 'bbox':[.1,.2,.7,.8]
    #     }]

    start_time = time.time()
    data = data[:500]
    batch_size = 500
    final_emb_results = []
    # modifed[Ammu]
    for i in tqdm(range(0, len(data), batch_size)):
    # find end of batch
        i_end = min(i+batch_size, len(data))
        data_batch = data[i:i_end]
        payload = parse_request(data_batch)
        payload_processed = _service.preprocess(payload)
        emb = _service.inference(payload_processed)
        emb = _service.postprocess(emb)
        final_emb_results = final_emb_results + emb
    end_time = time.time()
    print('Duration: ', end_time-start_time)
    # This print is for testing purpose
    print(emb)
    # modifed[Ammu]
    file_path = "embedding_data_output.json"
    # Writing data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(final_emb_results, json_file, indent=4)
