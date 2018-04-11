 #-*- coding: utf-8 -*-
__author__ = 'Youngki Baik'

import math

flow_thresh = 1e9
abs_thresh = 1.0
rel_thresh = 0.05


def is_valid(u, v):
    if u > flow_thresh:
        return False
    if v > flow_thresh:
        return False
    mag = math.sqrt(u*u + v*v)
    if mag < 1.0:
        return False
    return True


def epe_rate(flow, flow_gt):

    [height, width] = flow_gt.shape[0:2]

    pixels = 0
    error_cnt = 0

    for y in range(0, height):
        for x in range(0, width):
            [u, v] = flow[y, x, :]

            [u_gt, v_gt] = flow_gt[y, x, :]

            if not is_valid(u_gt, v_gt):
                continue

            pixels += 1

            u_diff = u_gt - u
            v_diff = v_gt - v
            dist = math.sqrt(u_diff**2 + v_diff**2)
            mag = math.sqrt(u**2 + v**2)

            if dist > abs_thresh and dist / mag > rel_thresh:
                error_cnt += 1

    error_rate = error_cnt / pixels
    return error_rate


def epe_average(flow, flow_gt):

    [height, width] = flow_gt.shape[0:2]

    pixels = 0
    dist_sum = 0.0

    for y in range(0, height):
        for x in range(0, width):
            [u, v] = flow[y, x, :]

            [u_gt, v_gt] = flow_gt[y, x, :]

            if not is_valid(u_gt, v_gt):
                continue

            pixels += 1

            u_diff = u_gt - u
            v_diff = v_gt - v
            dist = math.sqrt(u_diff**2 + v_diff**2)
            dist_sum += dist

    average_error = dist_sum / pixels
    return average_error
