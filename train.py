import os
import gc
import time
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from dataset import Meter
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.shedule import FixLR
from util.tool import collate_fn

import torch.distributed as dist

import os
import torch.multiprocessing as mp
from util.converter import keys,StringLabelConverter

lr = None
train_step = 0
converter=StringLabelConverter(keys)

def save_model(model, epoch, lr, optimzer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'textgraph_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()


    for i, (img,pointer_mask, dail_mask, text_mask, train_mask, transcripts, bboxs, mapping) in enumerate(train_loader):


        data_time.update(time.time() - end)

        train_step += 1

        img, pointer_mask, dail_mask, text_mask, train_mask = to_device(img, pointer_mask, dail_mask, text_mask, train_mask)

        output,pred_recog = model(img,bboxs,mapping) #4*12*640*640

        labels, label_lengths = converter.encode(transcripts.tolist())
        labels = to_device(labels)
        label_lengths = to_device(label_lengths)
        recog = (labels, label_lengths)


        loss_pointer, loss_dail, loss_text, loss_rec = criterion(output, pointer_mask, dail_mask, text_mask,train_mask, recog,pred_recog )

        loss = loss_pointer + loss_dail + loss_text + loss_rec

        # backward
        try:
            optimizer.zero_grad()
            loss.backward()
        except:
            print("loss gg")
            continue

        optimizer.step()
        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        gc.collect()


        if  i % cfg.display_freq == 0:
            print('({:d} / {:d})  Loss: {:.4f}  pointer_loss: {:.4f}  dail_loss: {:.4f}  text_loss: {:.4f} rec_loss: {:.4f}   '
                  .format(i, len(train_loader), loss.item(),loss_pointer.item(),loss_dail.item(), loss_text.item(), loss_rec.item()))


    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))
    


def main():
    global lr
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Meter(transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                    shuffle=True, num_workers=cfg.num_workers, pin_memory=True,collate_fn=collate_fn)

    # Model
    model = TextNet(backbone=cfg.net, is_training=True)

    model = model.to(cfg.device)

    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)

    criterion = TextLoss()

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

    print('Start training TextGraph_welcomeMEddpnew::--')
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.max_epoch+1):
        scheduler.step()
        train(model, train_loader, criterion, scheduler, optimizer, epoch)  #train
    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)

    # parse arguments
    option = BaseOptions()

    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # main
    main()

