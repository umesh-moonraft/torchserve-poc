from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2

# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = "/home/desktop/Documents/torchserve-poc/detectron/output/output_2/model_final.pth"
cfg.MODEL.DEVICE = 'cpu'

# Create predictor instance
predictor = DefaultPredictor(cfg)

# Load image
#/home/desktop/Documents/vera-ml-torchserve/mr-training/train-detectron2/training-dataset/curtain-dataset/images/test/7e8ab8ac36790952.jpg
image = cv2.imread("/home/desktop/Documents/torchserve-poc/detectron/test(1).jpg")

# Perform prediction
outputs = predictor(image)

threshold = 0.5

# Display predictions
preds = outputs["instances"].pred_classes.tolist()
scores = outputs["instances"].scores.tolist()
bboxes = outputs["instances"].pred_boxes

print(outputs["instances"])

# print(preds)

for j, bbox in enumerate(bboxes):
    bbox = bbox.tolist()

    score = scores[j]
    pred = preds[j]

    if score > threshold:
        x1, y1, x2, y2 = [int(i) for i in bbox]
        print(x1, y1, x2, y2, "-------------")
        print('Class Id: ', pred)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

cv2.imshow('image', image)
cv2.waitKey(0)
