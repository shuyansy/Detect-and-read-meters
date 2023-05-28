import torch
import numpy as np

def collate_fn(batch):
    img,  pointer_mask, dail_mask, text_mask, train_mask,boxes,transcripts = zip(*batch)
    bs = len(img)
    images = []
    pointer_maps = []
    dail_maps = []
    text_maps=[]
    training_masks = []

    for i in range(bs):
        if img[i] is not None:
            a = torch.from_numpy(img[i]).float()
            images.append(a)
            b = torch.from_numpy(pointer_mask[i]).long()
            pointer_maps.append(b)
            c = torch.from_numpy(dail_mask[i]).long()
            dail_maps.append(c)
            d = torch.from_numpy(text_mask[i]).long()
            text_maps.append(d)
            e = torch.from_numpy(train_mask[i])
            training_masks.append(e)

    images = torch.stack(images, 0)
    pointer_maps = torch.stack(pointer_maps, 0)
    dail_maps = torch.stack(dail_maps, 0)
    text_maps = torch.stack(text_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    mapping = []
    texts = []
    bboxs = []
    for index, gt in enumerate(zip(transcripts, boxes)):
        for t, b in zip(gt[0], gt[1]):
            mapping.append(index)
            texts.append(t)
            bboxs.append(b)


    mapping = np.array(mapping)
    texts = np.array(texts)
    bboxs = np.stack(bboxs, axis=0)
    bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis=1).astype(np.float32)

    return  images,pointer_maps, dail_maps, text_maps, training_masks, texts, bboxs, mapping

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl])