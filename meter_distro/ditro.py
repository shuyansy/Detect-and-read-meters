import math
import os
import cv2
import numpy as np
import torch
from meter_distro.models.net import U2NET
import matplotlib.pyplot as plt
from sympy import *
from skimage import morphology,data,color


class MeterReader_distro(object):

    def __init__(self, is_cuda=False):
        self.net = U2NET(3, 1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load('meter_distro/weight/distro_net.pt', map_location='cpu'))
        self.net.eval().to(self.device)

        self.threshold = 0.5  # 分割的阈值
        self.st=None

    @torch.no_grad()
    def __call__(self, image, image_name):
        image = self.square_picture(image, 416)
        image_tensor = self.to_tensor(image.copy()).to(self.device)
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().numpy()
        meter_mask=self.binary_image(mask[0])
        ori_image,cnt_image,cnt=self.read_image(image,meter_mask)
        #
        # cv2.imshow("img", image)
        # cv2.imshow("src", meter_mask)
        # cv2.imshow("src1", cnt_image)
        # cv2.waitKey(0)
        # print("cnt",len(cnt))

        if len(cnt) ==1:
            c=cnt[0]
        else:
            c=cnt[1]
        _, ap = self.meter_shape(c)


        if self.st == 'ellipse':
            print("ellipse")
            dst, circle_center = self.ellipse(c, cnt_image, ori_image)

        if self.st == 'rect':
            print("rect")
            dst,circle_center = self.rect(ori_image, ap)

        self.visulization(ori_image, dst, image_name)


        return dst,circle_center

    def meter_shape(self,contour):
        ep = 0.01 * cv2.arcLength(contour, True)
        ap = cv2.approxPolyDP(contour, ep, True)
        co = len(ap)
        # print("co",co)
        if co >=8:
            self.st = 'ellipse'
        else:
            self.st = 'rect'

        return co, ap

    def binary_image(self, image):
        """图片二值化"""
        condition = image > self.threshold
        image[condition] = 255
        image[~condition] = 0

        return image

    def read_image(self,ori_img, mask_img):

        mask_img = mask_img.astype(np.uint8)
        r, b = cv2.threshold(mask_img, 0, 255, cv2.THRESH_OTSU)
        a=cv2.Canny(b,0,255)
        cont_image, cnt, t = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return ori_img, cont_image, cnt

    def get_final_point(self,image):
        h, w, _ = image.shape
        if self.st == "ellipse":
            if w <= h:
                c4 = [0, 0.5 * h]
                c2 = [w, 0.5 * h]
                c3 = [0.5 * w, 0.5 * h + 0.5 * w]
                c1 = [0.5 * w, 0.5 * h - 0.5 * w]
            else:
                c4 = [0.5 * w - 0.5 * h, 0.5 * h]
                c2 = [0.5 * w + 0.5 * h, 0.5 * h]
                c3 = [0.5 * w, h]
                c1 = [0.5 * w, 0]

            final_p = [c1, c2, c3, c4]
            final_p = np.array(final_p)
            final_p = final_p.astype(np.float32)


        else:
            c1 = [0, 0]
            c4 = [0, h]
            c3 = [w, h]
            c2 = [w, 0]
            final_p = [c1, c2, c3, c4]
            final_p = np.array(final_p)
            final_p = final_p.astype(np.float32)

        return final_p, w, h

    def rect(self, ori_image, ap):

        ap=ap.reshape(-1,2)

        rec=cv2.minAreaRect(ap)
        original_p=cv2.boxPoints(rec)


        # original_p = ap.reshape(4, 2)
        original_p = original_p.astype(np.float32)
        original_p=self.order_points(original_p)
        final_p, w, h = self.get_final_point(ori_image)

        # print("original_p",original_p)
        # print("final_p", final_p)


        M = cv2.getPerspectiveTransform(original_p, final_p)
        dst = cv2.warpPerspective(ori_image, M, (w, h))

        center=(w,h)

        return dst,center


    def ellipse(self,contour, con_image, ori_image):
        retval = cv2.fitEllipse(contour)  # 取其中一个轮廓拟合椭圆
        img = cv2.ellipse(con_image, retval, (255, 0, 255), thickness=2)  # 在原图画椭圆


        center = retval[0]  # center point
        size = retval[1]  # 2b, 2a
        angle = retval[2]  # rotated angle
        # print(retval)
        # print(center)
        # print(size)
        # print("angle",angle)

        # 画长宽
        res_ellipse = (center, size, angle)
        ell_center_x = int(res_ellipse[0][0])
        ell_center_y = int(res_ellipse[0][1])

        ell_h_point1_x = int(ell_center_x - 0.5 * res_ellipse[1][0] * math.cos(res_ellipse[2] / 180 * math.pi))
        ell_h_point1_y = int(ell_center_y - 0.5 * res_ellipse[1][0] * math.sin(res_ellipse[2] / 180 * math.pi))
        ell_h_point2_x = int(ell_center_x + 0.5 * res_ellipse[1][0] * math.cos(res_ellipse[2] / 180 * math.pi))
        ell_h_point2_y = int(ell_center_y + 0.5 * res_ellipse[1][0] * math.sin(res_ellipse[2] / 180 * math.pi))

        ell_w_point1_x = int(ell_center_x - 0.5 * res_ellipse[1][1] * math.sin(res_ellipse[2] / 180 * math.pi))
        ell_w_point1_y = int(ell_center_y + 0.5 * res_ellipse[1][1] * math.cos(res_ellipse[2] / 180 * math.pi))
        ell_w_point2_x = int(ell_center_x + 0.5 * res_ellipse[1][1] * math.sin(res_ellipse[2] / 180 * math.pi))
        ell_w_point2_y = int(ell_center_y - 0.5 * res_ellipse[1][1] * math.cos(res_ellipse[2] / 180 * math.pi))

        cv2.line(img, (ell_h_point1_x, ell_h_point1_y), (ell_h_point2_x, ell_h_point2_y), (255, 0, 255), thickness=2)
        cv2.line(img, (ell_w_point1_x, ell_w_point1_y), (ell_w_point2_x, ell_w_point2_y), (255, 0, 255), thickness=2)



        if angle <= 90:
            p1 = [ell_w_point1_x, ell_w_point1_y]
            p2 = [ell_w_point2_x, ell_w_point2_y]
            p3 = [ell_h_point2_x, ell_h_point2_y]
            p4 = [ell_h_point1_x, ell_h_point1_y]

        else:
            p1 = [ell_h_point2_x, ell_h_point2_y]
            p2 = [ell_h_point1_x, ell_h_point1_y]
            p3 = [ell_w_point2_x, ell_w_point2_y]
            p4 = [ell_w_point1_x, ell_w_point1_y]

        original_p = [p1, p2, p3, p4]
        original_p = np.array(original_p)
        original_p = original_p.astype(np.float32)
        original_p=self.order_points(original_p)

        (x1, y1), radius = cv2.minEnclosingCircle(contour)
        # print("******",(x1, y1), radius)

        # 换成整数integer
        center = (int(x1), int(y1))
        radius = int(radius)
        # 画圆
        cv2.circle(img, center, radius, (255, 0, 0), 2)


        x = Symbol('x')
        y = Symbol('y')
        solved_value = solve([(x - x1) ** 2 + (y - y1) ** 2 - radius * radius, (x - original_p[0][0]) * (
                    (original_p[2][1] - original_p[0][1]) / (original_p[2][0] - original_p[0][0])) - y + original_p[0][
                                  1]], [x, y])
        # print("solved",solved_value)
        dis1 = (solved_value[0][0] - original_p[0][0]) ** 2 + (solved_value[0][1] - original_p[0][1]) ** 2
        dis2 = (solved_value[1][0] - original_p[0][0]) ** 2 + (solved_value[1][1] - original_p[0][1]) ** 2

        if dis1 <= dis2:
            p1 = list(solved_value[0])
            p3 = list(solved_value[1])
        else:
            p3 = list(solved_value[0])
            p1 = list(solved_value[1])

        # print("p1p3",p1,p3)

        x = Symbol('x')
        y = Symbol('y')
        solved_value1 = solve([(x - x1) ** 2 + (y - y1) ** 2 - radius * radius, (x - original_p[1][0]) * (
                    (original_p[3][1] - original_p[1][1]) / (original_p[3][0] - original_p[1][0])) - y + original_p[1][
                                   1]],
                              [x, y])
        # print("solved1", solved_value1)

        dis1 = (solved_value1[0][0] - original_p[1][0]) ** 2 + (solved_value1[0][1] - original_p[1][1]) ** 2
        dis2 = (solved_value1[1][0] - original_p[1][0]) ** 2 + (solved_value1[1][1] - original_p[1][1]) ** 2

        if dis1 <= dis2:
            p2 = list(solved_value1[0])
            p4 = list(solved_value1[1])
        else:
            p4 = list(solved_value1[0])
            p2 = list(solved_value1[1])

        h, w, _ = ori_image.shape
        final_p = [p1, p2, p3, p4]
        final_p = np.array(final_p)
        final_p = final_p.astype(np.float32)

        # print("original_p", original_p)
        # print("final_p", final_p)


        M = cv2.getPerspectiveTransform(original_p, final_p)
        dst = cv2.warpPerspective(ori_image, M, (w, h))

        return dst,(x1,y1)



    def order_points(self,pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        if leftMost[0, 1] != leftMost[1, 1]:
            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        else:
            leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
        (tl, bl) = leftMost
        if rightMost[0, 1] != rightMost[1, 1]:
            rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        else:
            rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
        (tr, br) = rightMost
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl])



    def visulization(self,ori, new,image_name):
        # plt.subplot(121)
        # plt.imshow(ori)
        # plt.title('ori')
        # plt.subplot(122)
        # plt.imshow(new)
        # plt.title('new')
        # plt.show()


        output = np.concatenate([ori, new], axis=1)
        cv2.imshow("src",output)
        cv2.waitKey(0)
        # name='result/'+image_name
        # cv2.imwrite(name,output)

        # cv2.imwrite('output/'+image_name,new)


    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        return background




if __name__ == '__main__':
    tester = MeterReader_distro()

    root = 'data/images/val1'
    for image_name in os.listdir(root):
        print(image_name)
        path = f'{root}/{image_name}'
        image = cv2.imread(path)
        restro_image, circle_center = tester(image,image_name)

