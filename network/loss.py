import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torch.nn import CTCLoss

class OCRLoss(nn.Module):

    def __init__(self):
        super(OCRLoss, self).__init__()
        self.ctc_loss = CTCLoss(zero_infinity=True)  # pred, pred_len, labels, labels_len

    def forward(self, *inputs):
        gt, pred = inputs[0], inputs[1]
        loss = self.ctc_loss(pred[0], gt[0], pred[1], gt[1])
        return loss


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.recogitionLoss = OCRLoss()


    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1

        input = input.contiguous().view(input.size()[0], -1)    #bs*hw
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d



    def forward(self, inputs, pointer_mask, dail_mask, text_mask,train_mask,y_true_recog, y_pred_recog):


        pointer_pred = inputs[:, 0]
        dail_pred = inputs[:, 1]
        text_pred=inputs[:,2]

        # modify tr_loss cross_entropy loss to dice loss

        loss_pointer = self.dice_loss(pointer_pred, pointer_mask, train_mask)
        loss_dail = self.dice_loss(dail_pred, dail_mask, train_mask)
        loss_text = self.dice_loss(text_pred, text_mask, train_mask)


        loss_pointer=loss_pointer.mean()
        loss_dail=loss_dail.mean()
        loss_text=loss_text.mean()

        recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)


        return loss_pointer,loss_dail,loss_text,recognition_loss

