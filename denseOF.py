 #-*- coding: utf-8 -*-
__author__ = 'Youngki Baik'

import sys
import cv2
from utils.flow_process import flow2bgr_middlebury
from utils.flow_io import read_flo, read_flow_kitti, write_flo
from utils.flow_eval import epe_rate, epe_average


def main(argv=sys.argv[1:]):

    if len(argv) != 2:
        print('Please enter "denseOF.py [current image] [next image]')
        #  print('Please enter "denseOF.py [current image] [next image] [ground truth.flo] ')
        return

    # load images
    img_curr = cv2.imread(argv[0], -1)
    img_next = cv2.imread(argv[1], -1)

    # color conversion
    if len(img_curr.shape) != 2:
        img_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)

    # do optical flow
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    #  optical_flow = cv2.FarnebackOpticalFlow_create()
    flow = optical_flow.calc(img_curr, img_next, None)

    # do coloring
    bgr = flow2bgr_middlebury(flow)
    cv2.imshow('flow', bgr)

    '''
    #  load ground truth data
    flow_gt = read_flo(argv[2])
    bgr_gt = flow2bgr_middlebury(flow_gt)
    cv2.imshow('flow_gt', bgr_gt)
    '''
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
