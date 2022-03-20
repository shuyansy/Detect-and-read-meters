import torch
import numpy as np
import cv2
import os
from util.config import config as cfg


#
# def visualize_gt(image, pointer=None, dail=None ,text=None) :
#     image_show = image.copy()
#     image_show = np.ascontiguousarray(image_show[:, :, ::-1])
#
#
#     if (tr is not None) and (tcl is not None) and (kernel is not None) and (border is not None):
#         tr = (tr > cfg.tr_thresh).astype(np.uint8)
#         tcl = (tcl > cfg.tcl_thresh).astype(np.uint8)
#         kernel = (kernel > cfg.tcl_thresh).astype(np.uint8)
#         border = (border > cfg.tcl_thresh).astype(np.uint8)
#         tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
#         tcl = cv2.cvtColor(tcl * 255, cv2.COLOR_GRAY2BGR)
#         kernel = cv2.cvtColor(kernel * 255, cv2.COLOR_GRAY2BGR)
#         border = cv2.cvtColor(border * 255, cv2.COLOR_GRAY2BGR)
#         image_show = np.concatenate([image_show, tr, tcl, kernel, border], axis=1)
#         return image_show
#     else:
#         return image_show


def visualize_detection(image, pointer=None, dail=None ,text=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    if (pointer is not None) and (dail is not None) and (text is not None):

        pointer = (pointer > cfg.pointer).astype(np.uint8)    #0.7
        dail = (dail >cfg.dail).astype(np.uint8)
        text = (text > cfg.text).astype(np.uint8)

        pointer = cv2.cvtColor(pointer * 255, cv2.COLOR_GRAY2BGR)
        dail = cv2.cvtColor(dail * 255, cv2.COLOR_GRAY2BGR)
        text = cv2.cvtColor(text * 255, cv2.COLOR_GRAY2BGR)


        image_show = np.concatenate([image_show, pointer, dail, text], axis=1)
        return image_show
    else:
        return image_show