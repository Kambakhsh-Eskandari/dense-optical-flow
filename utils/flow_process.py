 #-*- coding: utf-8 -*-
__author__ = 'Youngki Baik'

import math
import numpy as np
import cv2

flow_thresh = 1e9


def is_valid(u, v):
    if u > flow_thresh:
        return False
    if v > flow_thresh:
        return False
    return True


def make_color_wheel():
    # color encoding scheme
    # adapted from the color circle idea described at
    # http://members.shaw.ca/quadibloc/other/colint.htm
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((ncols, 3), dtype=np.int)  # r g b
    col = 0

    # RY
    color_wheel[0:RY, 0] = 255
    for i in range(0, RY):
        color_wheel[i, 1] = math.floor(  255*i / RY)
    col = col+RY

    # YG
    for i in range(0, YG):
        color_wheel[col+i, 0] = 255 - math.floor(255*i / YG)
        color_wheel[col:col+YG, 1] = 255
    col = col+YG

    # GC
    color_wheel[col:col+GC, 1] = 255
    for i in range(0, GC):
        color_wheel[col+i, 2] = math.floor(255*i / GC)
    col = col+GC

    # CB
    for i in range(0, CB):
        color_wheel[col+i, 1] = 255 - math.floor(255*i / CB)
        color_wheel[col:col+CB, 2] = 255
    col = col+CB

    # BM
    color_wheel[col:col+BM, 2] = 255
    for i in range(0, BM):
        color_wheel[col+i, 0] = math.floor(255*i / BM)
    col = col+BM

    # MR
    for i in range(0, MR):
        color_wheel[col+i, 2] = 255 - math.floor(255*i / MR)
        color_wheel[col:col+MR, 0] = 255

    return color_wheel


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


def flow2bgr(flow):
    bgr = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(bgr)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def interpolate_background(flow):

    [height, width] = flow.shape[0:2]
    flow_new = flow.copy()

    # for each row do
    for y in range(0, height):
        # init counter
        count = 0
        # for each pixel do
        for x in range(0, width):
            [u, v] = flow[y, x]
            # if flow is valid
            if is_valid(u, v):
                # at least one pixel requires interpolation
                if count > 0:
                    # count = max([count, 1])
                    x1 = x-count
                    x2 = x-1

                    # set pixel to min flow
                    if x1 > 0 and x2 < width-1:
                        fu_ipol = min(flow[y, x1 - 1, 0], flow[y, x2 + 1, 0])
                        fv_ipol = min(flow[y, x1 - 1, 1], flow[y, x2 + 1, 1])

                        for x_curr in range(x1, x2+1):
                            flow_new[y, x_curr, 0] = fu_ipol
                            flow_new[y, x_curr, 1] = fv_ipol
                     # reset counter
                    count = 0
            else:
                count += 1
    '''
        # extrapolate to the left
        for x in range (0, width):
            [u, v] = flow[y, x]
            if is_valid(u, v):
                for x2 in range (0, x):
                    flow_new[y, x2, :] = flow[y, x, :]
                break

        # extrapolate to the right
        for x in range (0, width):
            x = width -1 -x
            [u, v] = flow[y, x]

            if is_valid(u, v):
                for x2 in range(x+1, width):
                    flow_new[y, x2, :] = flow[y, x, :]
                break

    # for each column do
    for x in range(0, width):
        # extrapolate to the top
        for y in range(0, height):
            [u, v] = flow[y, x]
            if is_valid(u, v):
                for y2 in range (0, y):
                    flow_new[y2, x, :] = flow[y, x, :]
                break

        # extrapolate to the bottom
        for y in range(0, height):
            y = height -1 -y
            [u, v] = flow[y, x]

            if is_valid(u, v):
                for y2 in range (y+1, height):
                    flow_new[y2, x, :] = flow[y, x, :]
                break
    '''
    return flow_new


