from lib.utils_vvd.utils import create_uuid, cv_rgb_imwrite, glob_recursively, plt_image_show
from lib.utils_vvd.utils import boxes_painter
from lib.utils_vvd import json_load
import sys
from pathlib import Path
sys.path.append('external_lib/mmclassification')
sys.path.append('external_lib')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.controller import Controller
from Vi_cA_13 import Vi_cA_13_I
from Vi_cA_13_II import Vi_cA_13_II
from tqdm import tqdm
import time


def carrier_detection(image_path, model):

    # load image
    img = cv2.imread(image_path, 1)

    ring_obj = Vi_cA_13_I(img)

    carrier_info = Vi_cA_13_II(img, ring_obj.center_x_y, ring_obj.circle_list[1][-1], ring_obj.circle_list[2][-1])

    # find bboxes of bad revit
    start = time.time()
    results = model.infer(img, ring_obj.circle_list, carrier_info)
    print(f"time: {time.time() - start}s")

    if len(results) > 0:
        img = boxes_painter(img, results, color=[255,255,255])
        plt_image_show(img)
    pass

if __name__ == '__main__':
    # load regressor
    model = Controller(checkpoint='model/latest.pth', config_file='model/mobilenet.py')
    image_path_list = glob_recursively('assets', 'png')

    for image_path in tqdm(image_path_list):
        carrier_detection(image_path, model)
    pass
