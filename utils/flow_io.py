 #-*- coding: utf-8 -*-
__author__ = 'Youngki Baik'

import struct
import numpy as np
import cv2

tag_string = b'PIEH'
UNKNOWN_FLOW = 1e10


def read_flo(filename):
    f = open(filename, "rb")
    tag = f.read(4)
    (width,) = struct.unpack('i', f.read(4))
    (height,) = struct.unpack('i', f.read(4))

    if tag != tag_string:
        print('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % filename)
        return
    if width < 1 or width > 99999:
        print('readFlowFile(%s): illegal width %d' % (filename, width))
        return
    if height < 1 or height > 99999:
        print('readFlowFile(%s): illegal height %d' % ( filename, height))
        return

    data = np.fromfile(f, np.float32)
    data = data.reshape(height, width, 2)

    f.close()
    return data


def write_flo(filename, flow):
    f = open(filename, "wb")
    [height, width] = flow.shape[0:2]

    f.write(tag_string)
    f.write(struct.pack("i", width))
    f.write(struct.pack("i", height))
    f.write(bytearray(flow))

    f.close()


def read_flow_kitti(path):
    img = cv2.imread(path, -1)
    rows, cols = img.shape[:2]
    flow = np.zeros((rows, cols, 2), np.float32)

    img = img.astype(np.float32)
    flow[..., 0] = (img[..., 2] - 32768.0) / 64.0
    flow[..., 1] = (img[..., 1] - 32768.0) / 64.0

    for row in range (0, rows):
        for col in range(0, cols):
            if img[row, col, 0] == 0:
                flow[row, col, 0] = UNKNOWN_FLOW
                flow[row, col, 1] = UNKNOWN_FLOW

    return flow


def write_flow_kitti(filename, flow):
    # flow 2 image
    height, width = flow.shape[:2]
    img = np.zeros((height, width, 3), np.uint16)

    img[..., 2] = np.maximum(np.minimum(flow[..., 0] * 64.0 + 32768.0, 65535.0), 0.0).astype(np.uint16)
    img[..., 1] = np.maximum(np.minimum(flow[..., 1] * 64.0 + 32768.0, 65535.0), 0.0).astype(np.uint16)
    img[..., 0] = 1

    cv2.imwrite(filename, img)

