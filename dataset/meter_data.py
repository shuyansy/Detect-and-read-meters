#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import json


class Meter(TextDataset):

    def __init__(self, root='./data', mode='train',mode1='train1',is_training=True,transform=None):
        super().__init__(transform, is_training)
        self.dataset = []
        self.name=[]
        image_path = f'{root}/images/'
        mask_path = f'{root}/annotations/{mode}'
        mask_path1 = f'{root}/annotations/{mode1}'

        for image_name in os.listdir(image_path):
            mask_name = image_name.split('.')[0] + '.json'
            self.dataset.append((f'{image_path}/{image_name}', f'{mask_path}/{mask_name}', f'{mask_path1}/{mask_name}'))
            self.name.append(image_name)


    @staticmethod
    def parse_txt(mask_path,mask_path1):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """

        with open(mask_path, 'r') as load_f:
            load_dict = json.load(load_f)

        info = load_dict['shapes']
        polygons = []
        for i in range(len(info)):
            points= info[i]['points']
            points = np.array(points).astype(np.int32)
            label=info[i]['label']
            polygons.append(TextInstance(points, 'c', label))


        with open(mask_path1, 'r') as load_f:
            load_dict = json.load(load_f)
        info = load_dict['shapes']

        transcripts=[]
        for i in range(len(info)):
            points= info[i]['points']
            points = np.array(points).astype(np.int32)
            text=info[i]['label']
            transcripts.append(text)
            label='number'
            polygons.append(TextInstance(points, 'c', label))


        return polygons,transcripts

    def __getitem__(self, item):
        image_path, mask_path, mask_path1 = self.dataset[item]
        idx=self.name[item]

        # Read image data
        image = pil_load_img(image_path)

        try:
            polygons,transcripts = self.parse_txt(mask_path,mask_path1)
        except:
            polygons = None

        # print("poltygons",polygons)

        return self.get_training_data(image, polygons,transcripts,image_id=idx, image_path=image_path)


    def __len__(self):
        return len(self.dataset)


