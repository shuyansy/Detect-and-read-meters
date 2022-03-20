import os
import cv2
import numpy as np
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# Load model
device='0'
weights='runs/train/exp6/weights/best.pt'
data='data/mydata.yaml'

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size([640], s=stride)  # check image size

path='/home/sy/ocr/datasets/all_meter_image/'
img_dir=os.listdir(path)
for i in img_dir:

    img=cv2.imread(path+i)
    # Padded resize
    img = letterbox(img, [640], stride=stride, auto=pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

