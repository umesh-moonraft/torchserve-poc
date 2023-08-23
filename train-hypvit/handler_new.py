

import torch
import utils
from bn_inception import *

import os


class ModelHandler(object):

    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.model = None
        self.model_file = "bn_inception-52deb4733.pth"
        self.device = False
        self.transform = utils.make_transform(is_train=False, is_inception=1)
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

        self.device = torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

        try:
            model_pt_path = self.model_file
            self.model = bn_inception(embedding_size=512, pretrained=True, is_norm=1, bn_freeze=1)

            # checkpoint = torch.load(model_pt_path)
            # self.model.load_state_dict(checkpoint['model_state_dict'])

            # if (self.device == "cpu"):
            #     print('CPU')
            #     self.model.load_state_dict(torch.load(
            #         model_pt_path, map_location="cpu"), strict=False)
            # else:
            #     print('GPU')
            #     checkpoint = torch.load(model_pt_path)
            #     # self.model.load_state_dict(checkpoint['model_state_dict'])
            #     self.model.load_state_dict(
            #         checkpoint['model_state_dict'], strict=False)

            print('Model file {0} loaded successfully'.format(model_pt_path))

            print("model built on initialize")
        except AssertionError as error:
            # Output expected AssertionErrors.
            print(error)
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            print("Error: {}".format(e))

        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True
        print("initialized")

               

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


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        print("service not initialized")
        _service.initialize(context)

    if data is None:
        return None
    

    # print("data ----> ", data)
    return _service.handle(data, context)
