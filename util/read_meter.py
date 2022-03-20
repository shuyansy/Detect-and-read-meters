import math
import os
import cv2
import numpy
import numpy as np
import torch
from skimage import morphology

class MeterReader(object):

    def __init__(self):
        pass


    def __call__(self, image,point_mask,dail_mask,word_mask,number,std_point):

        img_result = image.copy()
        value=self.find_lines(img_result,point_mask,dail_mask,number,std_point)


        return value




    def find_lines(self,ori_img,pointer_mask,dail_mask,number,std_point):

        # 实施骨架算法
        pointer_skeleton = morphology.skeletonize(pointer_mask)
        pointer_edges = pointer_skeleton * 255
        pointer_edges = pointer_edges.astype(np.uint8)
        # cv2.imshow("pointer_edges", pointer_edges)
        # cv2.waitKey(0)

        dail_mask = np.clip(dail_mask, 0, 1)
        dail_edges = dail_mask * 255
        dail_edges = dail_edges.astype(np.uint8)
        # cv2.imshow("dail_edges", dail_edges)
        # cv2.waitKey(0)

        pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10,
                                        maxLineGap=400)
        coin1, coin2 = None, None
        for x1, y1, x2, y2 in pointer_lines[0]:
            coin1 = (x1, y1)
            coin2 = (x2, y2)
            cv2.line(ori_img, (x1, y1), (x2, y2), (255, 0, 255), 2)


        h, w, _ = ori_img.shape
        center = (0.5 * w, 0.5 * h)
        dis1 = (coin1[0] - center[0]) ** 2 + (coin1[1] - center[1]) ** 2
        dis2 = (coin2[0] - center[1]) ** 2 + (coin2[1] - center[1]) ** 2
        if dis1 <= dis2:
            pointer_line = (coin1, coin2)
        else:
            pointer_line = (coin2, coin1)

        # print("pointer_line", pointer_line)

        if std_point==None:
            return "can not detect dail"

        # calculate angle
        a1 = std_point[0]
        a2 = std_point[1]
        one = [[pointer_line[0][0], pointer_line[0][1]], [a1[0], a1[1]]]
        two = [[pointer_line[0][0], pointer_line[0][1]], [a2[0], a2[1]]]
        three = [[pointer_line[0][0], pointer_line[0][1]], [pointer_line[1][0], pointer_line[1][1]]]
        # print("one", one)
        # print("two", two)
        # print("three",three)

        one=np.array(one)
        two=np.array(two)
        three = np.array(three)

        v1=one[1]-one[0]
        v2=two[1]-two[0]
        v3 = three[1] - three[0]

        distance=self.get_distance_point2line([a1[0], a1[1]],[pointer_line[0][0], pointer_line[0][1], pointer_line[1][0], pointer_line[1][1]])
        # print("dis",distance)

        flag=self.judge(pointer_line[0],std_point[0],pointer_line[1])
        # print("flag",flag)

        std_ang = self.angle(v1, v2)
        # print("std_result", std_ang)
        now_ang = self.angle(v1, v3)
        if flag >0:
            now_ang=360-now_ang
        # print("now_result", now_ang)


        # calculate value
        if number!=None and number[0]!="":
            two_value = float(number[0])
        else:
            return "can not recognize number"
        if std_ang * now_ang !=0:
            value = (two_value / std_ang)
            value=value*now_ang
        else:
            return "angle detect error"

        if flag>0 and distance<40:
            value=0.00
        else:
            value=round(value,3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        ori_img = cv2.putText(ori_img, str(value), (30, 30), font, 1.2, (255, 0 , 255), 2)

        cv2.imshow("result",ori_img)
        cv2.waitKey(0)

        return value

    def get_distance_point2line(self, point, line):
        """
        Args:
            point: [x0, y0]
            line: [x1, y1, x2, y2]
        """
        line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance


    def judge(self,p1,p2,p3):
        A=p2[1]-p1[1]
        B=p1[0]-p2[0]
        C=p2[0]*p1[1] - p1[0]*p2[1]

        value=A*p3[0] + B*p3[1] +C

        return value


    def angle(self,v1, v2):
        lx=np.sqrt(v1.dot(v1))
        ly=np.sqrt(v2.dot(v2))
        cos_angle=v1.dot(v2) / (lx * ly)

        angle=np.arccos(cos_angle)
        angle2=angle*360 / 2 / np.pi

        return angle2



if __name__ == '__main__':
    tester = MeterReader()
    root = 'data/images/val'
    for image_name in os.listdir(root):
        print(image_name)
        path = f'{root}/{image_name}'
        image = cv2.imread(path)
        result = tester(image)
        print(result)
        # cv2.imshow('a', image)
        # cv2.waitKey()
