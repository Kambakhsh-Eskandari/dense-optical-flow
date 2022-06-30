 #-*- coding: utf-8 -*-
__author__ = 'Youngki Baik'

import sys
import cv2
#from utils.flow_process import flow2bgr_middlebury
from utils.flow_io import read_flo, read_flow_kitti, write_flo
from utils.flow_eval import epe_rate, epe_average

def flow2bgr_middlebury(flow):
    color_wheel = make_color_wheel()
    [ncols, channel] = color_wheel.shape

    [height, width] = flow.shape[0:2]
    bgr = np.zeros((height, width, 3), dtype=np.uint8)

    # compute max magnitude
    u = flow[:, :, 0].copy()
    v = flow[:, :, 1].copy()
    u[u > flow_thresh] = 0
    v[v > flow_thresh] = 0

    mag = np.sqrt(u**2 + v**2)
    max_mag = mag.max()
    #max_mag = (u.max()+ v.max())*2+1

    if max_mag == 0:
        return bgr

    scale = 1.0/max_mag

    # scaling
    u = u * scale
    v = v * scale
    mag = np.sqrt(u ** 2 + v ** 2)

    for y in range(0, height):
        for x in range(0, width):

            [ud, vd] = flow[y, x]
            if not is_valid(ud, vd):
                continue

            rad = mag[y, x]
            a = np.arctan2(-v[y, x], -u[y, x]) / np.pi
            fk = (a + 1) / 2 * (ncols - 1)
            k0 = int(fk)
            k1 = (k0 + 1) % ncols
            f = fk - k0

            for c in range (0, channel):
                col0 = float(color_wheel[k0, c])/255.0
                col1 = float(color_wheel[k1, c])/255.0
                col = (1 - f) * col0 + f * col1

                if rad <= 1:
                    col = 1 - rad * ( 1 - col)
                else:
                    col *= 0.75
                bgr[y, x, 2-c] = int(255.0*col)

    return bgr


img_curr = cv2.imread('Resources\img1L.png')
img_next = cv2.imread('Resources\img2L.png')


img_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)


optical_flow = cv2.DualTVL1OpticalFlow_create()

flow = optical_flow.calc(img_curr, img_next, None)


bgr = flow2bgr_middlebury(flow)
cv2.imshow('flow', bgr)

'''
    #  load ground truth data
    flow_gt = read_flo(argv[2])
    bgr_gt = flow2bgr_middlebury(flow_gt)
    cv2.imshow('flow_gt', bgr_gt)
'''
cv2.waitKey(0)



