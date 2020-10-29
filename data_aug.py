#!/usr/env/bin python3
import glob
import os
import random

import cv2

import numpy as np
import hashlib
import sys


def prob(percent):
    """
    percent: 0 ~ 1, e.g: 如果 percent=0.1，有 10% 的可能性
    """
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


def apply_blur_on_output(img):
    if prob(0.5):
        return apply_gauss_blur(img, [3, 5])
    else:
        return apply_norm_blur(img)

def apply_gauss_blur(img, ks=None):
    if ks is None:
        ks = [7, 9, 11, 13]
    ksize = random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize <= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img

def apply_norm_blur(img, ks=None):
    # kernel == 1, the output image will be the same
    if ks is None:
        ks = [2, 3]
    kernel = random.choice(ks)
    img = cv2.blur(img, (kernel, kernel))
    return img

def apply_prydown(img):
    """
    模糊图像，模拟小图片放大的效果
    """
    scale = random.uniform(1, 1.5)
    height = img.shape[0]
    width = img.shape[1]

    out = cv2.resize(img, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)

def apply_lr_motion(image):    
    kernel_size = 5
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image


def apply_up_motion(image): 
    kernel_size = 9
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image
