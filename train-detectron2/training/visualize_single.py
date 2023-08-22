import os
import random

import matplotlib.pyplot as plt
import cv2



fig = plt.figure()
img = cv2.imread("/home/desktop/Documents/torchserve-poc/train-detectron2/chair.jpg")

h_img, w_img, _ = img.shape
lines = [
        [
            223.45111083984375,
            67.86144256591797,
            797.518798828125,
            943.943603515625
        ],
        # [
            
        #     27.574119567871094,
        #     67.69054412841797,
        #     478.41375732421875,
        #     858.2264404296875
        # ],
        # [
            
        #     417.8492736816406,
        #     338.13507080078125,
        #     512.643310546875,
        #     453.03338623046875
        # ],
        # [
            
        #     221.95013427734375,
        #     67.28851318359375,
        #     782.3932495117188,
        #     929.4290161132812
        # ],
        # [
            
        #     265.98583984375,
        #     147.1990966796875,
        #     592.993408203125,
        #     577.220947265625
        # ],
        # [
            
        #     265.7705078125,
        #     367.96990966796875,
        #     460.0799560546875,
        #     831.4443969726562
        # ],
        # [
            
        #     507.3913879394531,
        #     109.72667694091797,
        #     984.57275390625,
        #     760.607421875
        # ],
        # [
            
        #     295.8254699707031,
        #     383.7369384765625,
        #     485.38873291015625,
        #     626.1302490234375
        # ]
]
for bbox in lines:
    # line = line.split(' ')
    x1, y1, x2, y2 = [int(i) for i in bbox]
    print(x1, y1, x2, y2, "-------------")
    # print('Class Id: ', pred)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# for line in lines:
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)


cv2.imshow('image', img)
cv2.waitKey(0)