import torch
import requests
import utils
import sys, io ,time
from os import path
from PIL import Image


from bn_inception import *


def get_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        image_bytes = response.content

        input = io.BytesIO(image_bytes)
        pil_image = Image.open(input).convert("RGB")

        # img = cv2.imdecode(np.fromstring(input.read(), np.uint8), 1)

        return {"image": pil_image}
    except Exception as e:
        print(e)
        return {"error": "Inavalid image"}


def validate_arguments(data):
    print("validate_arguments data-->", data)
    bbox = data.get("bbox", None)
    image_url = data.get("image_url", None)

    if bbox is None or image_url is None:
        return {"error": "Inavalid arguments pass bbox and image_url"}

    # download image
    image_res = get_image(image_url)

    if "error" in image_res:
        return {**data, **image_res}

    image = image_res["image"]

    bbox_res = validate_bbox(bbox, image)

    return bbox_res


def parse_request(request):
    # each item in the list is a dictionary with a single body key, get the body of the request
    request_body = request.get("body")
    request_instances = request_body.get("instances")

    payloads = []

    for image_instance in request_instances:
        payload = validate_arguments(image_instance)

        payloads.append(payload)

    return payloads


def validate_bbox(bbox, image):
    # check if the image bbox values are normalized
    bbox = eval(bbox)
    # length bbox check
    bbox_sanity_check = True if len(bbox) == 4 else False

    print(image, image.size)

    img_width, img_height = image.size

    print(img_width, img_height)

    payload = {}

    if bbox_sanity_check:
        # detectron2 bbox are not normalized
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > img_width or bbox[3] > img_height:
            bbox_sanity_check = False
        else:
            bbox_processed = bbox

        try:
            # cropped image
            cropped_image = image.crop((bbox_processed))
            # check if the cropped image is valid or not
            # this also ensures if the crop is logically wrong or outside the image
            if cropped_image.tobytes():
                bbox_sanity_check = True

        except Exception as e:
            print("error in bbox validation", e)
            bbox_sanity_check = False

    if bbox_sanity_check:
        payload = {"image_cropped": cropped_image, "bbox": bbox}
    else:
        payload = {
            "error": "bbox should be list of absolute values with the format [x1,y1,x2,y2]"
        }

    return payload


class ModelHandler(object):

    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self._gpu_id = 0
        self.initialized = False
        self.model = None
        self.model_file = "bn_inception-52deb4733.pth"
        # self.model_file = "/Users/apple/Desktop/Vera_project_files/torchserve-poc/train-hypvit/output/bn_inception-52deb4733.pth"

        self.device = False
        self.transform = utils.make_transform(is_train=False, is_inception=1)
        self.config_file = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        print("initializing starting")

        print(
            "File {} exists {}".format(
                self.model_file, str(path.exists(self.model_file))
            )
        )
        # print(
        #     "File {} exists {}".format(
        #         self.config_file, str(path.exists(self.config_file))
        #     )
        # )

        self._gpu_id = context.system_properties["gpu_id"]

        print("_gpu_id == ", self._gpu_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model_pt_path = self.model_file
            self.model = bn_inception(
                embedding_size=512,
                pretrained=True,
                is_norm=1,
                bn_freeze=1,
                local_weight_path=self.model_file,
            )

            self.model.load_state_dict(
                torch.load(model_pt_path, map_location=self.device), strict=False
            )

            print("Model file {0} loaded successfully".format(model_pt_path))

            print("model built on initialize")
        except AssertionError as error:
            # Output expected AssertionErrors.
            print(error)
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            print("Error: {}".format(e))

        self._context = context
        # uncomment for torchserve
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
        print("_batch_size == ", self._batch_size)

        payloads = []

        # batch is a list of requests
        for request in batch:
            instance_payloads = parse_request(request)
            payloads += instance_payloads

        print("pre-processing finished for a batch of {}".format(len(batch)))

        return payloads

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # [{'image_cropped': <PIL.Image.Image image mode=RGB size=98x358 at 0x134811450>, 'bbox': [94.83322143554688, 17.79736328125, 193.27203369140625, 376.315185546875]}]
        # Do some inference call to engine here and return output
        print("inference started for a batch of {}".format(len(model_input)))

        local_payloads = []
        outputs = []

        for payload in model_input:
            
            # generate bbox for image
            if "error" not in payload:
                input_img = payload['image_cropped']
                pil_processed_image =  self.transform(input_img).unsqueeze(0)
                local_payloads.append(pil_processed_image)

        if len(local_payloads):
            outputs = self.model(torch.cat(local_payloads))
            outputs = list(torch.split(outputs, 1))

        output_payloads = []
        for payload in model_input:
            if "error" in payload:
                output_payloads.append(payload)
            else:
                output_payloads.append({"instances": outputs.pop(0)})

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
        
        final_data = []
        for payload in inference_output:
            if "error" in payload:
                final_data.append(payload)
            else:
                instances = payload["instances"]
                result = instances.detach().cpu().numpy().squeeze()
                data = {"embedding": {"data": result.tolist(), "version": 5}}
                final_data.append(data)
        elapsed_time = time.time() - start_time
            
        print("post-processing finished for a batch of {} in {}".format(len(inference_output), elapsed_time))

        return final_data
    

    
    def postprocess(self, payloads):
        final_data = []
        for payload in payloads:
            if "error" in payload:
                final_data.append(payload)
            else:
                instances = payload["instances"]
                result = instances.detach().cpu().numpy().squeeze()
                data = {"embedding": {"data": result.tolist(), "version": 5}}
                final_data.append(data)
        return final_data

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

        print("model_input ----> ")
        print("model_input ----> ", model_input)
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

    # print("data ----> ", data)
    return _service.handle(data, context)


# #### THIS IS FOR RUNNING LOCALLY
if __name__ == "__main__":
    context = {"system_properties": {"batch_size": 1}}

    if not _service.initialized:
        _service.initialize(context)

    data = {
        "instances": [
            {
                "image_url": "https://m.media-amazon.com/images/I/81ZQXAE1OVL._AC_UL400_.jpg",
                "bbox": "[94.83322143554688,17.79736328125,193.27203369140625,376.315185546875]",
            }
        ]
    }

    data = [{"body": data}]

    output = _service.handle(data, context)
    print("output------>", output)
