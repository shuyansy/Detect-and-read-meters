import os
import cv2
import numpy as np
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
import torch
from util.misc import to_device
from util.read_meter import MeterReader
from util.converter import keys,StringLabelConverter
from get_meter_area import  Detector

# parse arguments

option = BaseOptions()
args = option.initialize()

update_config(cfg, args)
print_config(cfg)

predict_dir='demo/'

model = TextNet(is_training=False, backbone=cfg.net)
model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                          'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
model.load_model(model_path)
model = model.to(cfg.device)
converter=StringLabelConverter(keys)

det=Detector()
detector = TextDetector_mask(model)
meter = MeterReader()
transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)


image_list=os.listdir(predict_dir)



for index in image_list:
    print("**************",index)
    image = cv2.imread(predict_dir+index)
    
    cv2.imshow("det1",image)
    cv2.waitKey(0)

    
    # detect meter area
    image, image_info, digital_list, meter_list = det.detect(image, index)

    if len(meter_list)==0:
        print("no detected meter")
        continue
    else:
        for i in meter_list:
            cv2.imshow("det",i)
            cv2.waitKey(0)

            image,_=transform(i)
            image = image.transpose(2, 0, 1)
            image=torch.from_numpy(image).unsqueeze(0)
            image=to_device(image)

            output = detector.detect1(image)

            pointer_pred, dail_pred, text_pred, preds, std_points = output['pointer'], output['dail'], output['text'], output['reco'], output['std']

            # decode predicted text
            pred, preds_size = preds
            if pred != None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
                # print("preds", pred_transcripts)
            else:
                pred_transcripts = None


            img_show = image[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)


            result = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points)

