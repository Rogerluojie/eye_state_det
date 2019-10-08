# -*- coding: UTF-8 -*-
# @Time      :
# @author    : Roger
# @software  : pyCharm Community Edition


import os

import sys,time
import math
caffe_root = "/home/eryuan/soft/caffe-mobilenet_senet/"
sys.path.insert(1,caffe_root + "python")
import caffe
import cv2
import numpy as np


# caffe.set_mode_cpu()
GPU_ID = 0  # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
net_24 = caffe.Net("../train/deploy.prototxt",
                   "../eye_model/solver_iter_100000.caffemodel",
                 caffe.TEST)

def detectFace_eye(img):
    """
    detect face by CNN
    :param img: the resized image mat
    :return: the output score of the mat in the CNN
    """
    net_24.blobs['X'].reshape(1, 3, 24, 24)
    img = cv2.resize(img, (24, 24))
    img = (img - 127.5) / 127.5

    img = img.transpose((2, 0, 1))
    net_24.blobs['X'].data[...] = img
    out = net_24.forward()
    cls_prob = out['prob']
    return cls_prob[0]
