import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.vgg import VggNet
from network.resnet import ResNet
from util.roi import batch_roi_transform
from network.crnn import CRNN
from util.converter import keys
from util.misc import mkdirs, to_device
import cv2
from util.tool import order_points

class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class FPN(nn.Module):

    def __init__(self, backbone='vgg_bn', is_training=True):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone
        self.class_channel = 6
        self.reg_channel = 2

        if backbone == "vgg" or backbone == 'vgg_bn':
            if backbone == 'vgg_bn':
                self.backbone = VggNet(name="vgg16_bn", pretrain=True)
            elif backbone == 'vgg':
                self.backbone = VggNet(name="vgg16", pretrain=True)

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            self.merge2 = UpBlok(128 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 32)

        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet101':
                self.backbone = ResNet(name="resnet101", pretrain=True)
            elif backbone == 'resnet50':
                self.backbone = ResNet(name="resnet50", pretrain=True)

            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 256)
            self.merge3 = UpBlok(512 + 256, 128)
            self.merge2 = UpBlok(256 + 128, 64)
            self.merge1 = UpBlok(64 + 64, 32)
        else:
            print("backbone is not support !")

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)

        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)

        return up1, up2, up3, up4, up5


class TextNet(nn.Module):
    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, self.is_training)


        # ##class and regression branch
        self.out_channel = 3
        self.predict = nn.Sequential(
            nn.Conv2d(32, self.out_channel, kernel_size=1, stride=1, padding=0)
        )

        num_class = len(keys) + 1
        self.recognizer = Recognizer(num_class)


    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x,boxes,mapping):
        up1, up2, up3, up4, up5 = self.fpn(x)
        predict_out = self.predict(up1)

        rois = batch_roi_transform(x, boxes[:, :8], mapping)
        pred_mapping = mapping
        pred_boxes = boxes

        # print("rois",rois.shape)
        preds = self.recognizer(rois)
        # print("preds",preds.shape)

        preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1)))
        preds_size=to_device(preds_size)
        # print("predsize", preds_size)

        return predict_out,(preds, preds_size)


    def forward_test(self, x):
        up1, up2, up3, up4, up5 = self.fpn(x)
        output = self.predict(up1)
        # print("predict_out",output.shape)

        pointer_pred = torch.sigmoid(output[0, 0, :, :]).data.cpu().numpy()
        dail_pred = torch.sigmoid(output[0, 1, :, :]).data.cpu().numpy()
        text_pred = torch.sigmoid(output[0, 2, :, :]).data.cpu().numpy()
        pointer_pred = (pointer_pred > 0.5).astype(np.uint8)
        dail_pred = (dail_pred > 0.5).astype(np.uint8)
        text_pred = (text_pred > 0.7).astype(np.uint8)
        
        

        dail_label=self.filter(dail_pred,n=30)
        text_label = self.filter(text_pred)
        
        
        # cv2.imshow("srtc",text_pred*255)
        # cv2.imshow("1", pointer_pred * 255)
        # cv2.imshow("2", dail_label * 255)
        # cv2.waitKey(0)

        # order dail_label by y_coordinates
        dail_edges = dail_label * 255
        dail_edges = dail_edges.astype(np.uint8)
        _, dail_contours,_ = cv2.findContours(dail_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # new
        text_edges = text_label * 255
        text_edges = text_edges.astype(np.uint8)
        _, text_contours,_ = cv2.findContours(text_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ref_point = []
        for i in range(len(text_contours)):
            rect = cv2.minAreaRect(text_contours[i])
            ref_point.append((int(rect[0][0]), int(rect[0][1])))
        # print("ref",ref_point)
        
        
        std_point = []
        for i in range(len(dail_contours)):
            rect = cv2.minAreaRect(dail_contours[i])
            std_point.append((int(rect[0][0]), int(rect[0][1])))
        
        
        # print("std",std_point)
        
        if len(std_point)==0:
            return pointer_pred, dail_label, text_label, (None, None),None
            
        
        if len(std_point) < 2:
                # std_point=None
            std_point.append(ref_point[0])
            # return pointer_pred, dail_label, text_label, (None, None),[std_point[0],ref_point[0]]
        else:
            if std_point[0][1] >= std_point[1][1]:
                pass
            else:
                std_point[0], std_point[1] = std_point[1], std_point[0]

        # print("******",std_point)

       

        word_edges =text_label* 255
        img_bin, contours, hierarchy = cv2.findContours(word_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_dis=10000
        index=0
        if len(contours) != 0:
            for i in range(len(contours)):
                min_rect = cv2.minAreaRect(contours[i])

                test_point=(min_rect[0][0], min_rect[0][1])
                dis=(test_point[0]-std_point[1][0]) **2 + (test_point[1]-std_point[1][1]) **2
                if dis<max_dis:
                    max_dis=dis
                    index=i

            rect_points = cv2.boxPoints(cv2.minAreaRect(contours[index]))
            bboxes = np.int0(rect_points)
            bboxes=order_points(bboxes)
            # print("bbox", bboxes)
            boxes=bboxes.reshape(1,8)
            mapping=[0]
            mapping=np.array(mapping)
            rois = batch_roi_transform(x,boxes[:, :8], mapping)
            # print("rois",rois.shape)
            preds = self.recognizer(rois)
            preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1)))
            # print("*******", preds.shape, preds_size)

        else:
            preds=None
            preds_size=None
        
        
        
       
        return pointer_pred,dail_label,text_label,(preds,preds_size),std_point


    def filter(self,image,n=30):
        text_num, text_label = cv2.connectedComponents(image.astype(np.uint8), connectivity=8)
        for i in range(1, text_num + 1):
            pts = np.where(text_label == i)
            if len(pts[0]) < n:
                text_label[pts] = 0
        text_label = text_label > 0
        text_label = np.clip(text_label, 0, 1)
        text_label = text_label.astype(np.uint8)
        return text_label






class Recognizer(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.crnn = CRNN(32, 1, nclass, 256)

    def forward(self, rois):
        return self.crnn(rois)


if __name__=="__main__":
    csrnet=TextNet().to('cuda')
    img=torch.ones((1,3,256,256)).to('cuda')
    out=csrnet(img)
    print(out.shape)        # 1*12*256*256

