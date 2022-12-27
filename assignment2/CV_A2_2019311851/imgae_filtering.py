import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import time


def cross_correlation_1d(img, kernel):  # 11,1 -> 1, 11
    img_h = img.shape[0]
    img_w = img.shape[1]
    if kernel.shape[0] != 1:  # 수평
        pad_size = kernel.shape[0] // 2
    else:  # 수직
        pad_size = kernel.shape[1] // 2
    img_size = len(img)

    # 패딩
    for i in range(pad_size):
        if kernel.shape[0] != 1:  # (11, 1)
            img = np.insert(img, img_h + i * 2, img[-1], axis=0)
            img = np.insert(img, 0, img[0], axis=0)
        else:  # (1, 11)
            img = np.insert(img, img_w + i * 2, img.T[-1], axis=1)
            img = np.insert(img, 0, img.T[0], axis=1)

    # filtering
    new_img = np.zeros((img_h, img_w), dtype=float)

    if kernel.shape[0] == 1:  # (1, 11)
        for i in range(0, img.shape[0]):
            for j in range(0 + pad_size, img.shape[1] - pad_size):
                new_img[i][j - pad_size] += sum(sum(kernel * img[i, j - pad_size:j + pad_size + 1]))
    else:  # # (11, 1)
        for i in range(0 + pad_size, img.shape[0] - pad_size):
            for j in range(0, img.shape[1]):
                new_img[i - pad_size][j] += sum(kernel * img[i - pad_size:i + pad_size + 1, j])

    return new_img


def cross_correlation_2d(img, kernel):
    # h, w = kernel.shape
    img_h = img.shape[0]
    img_w = img.shape[1]
    pad_size = kernel.shape[0] // 2

    # 패딩
    for i in range(pad_size):
        img = np.insert(img, img_h + i * 2, img[-1], axis=0)
        img = np.insert(img, 0, img[0], axis=0)
        img = np.insert(img, img_w + i * 2, img.T[-1], axis=1)
        img = np.insert(img, 0, img.T[0], axis=1)

    # filtering 계산
    new_img = np.zeros((img_h, img_w), dtype=float)

    for i in range(0 + pad_size, img.shape[0] - pad_size):
        for j in range(0 + pad_size, img.shape[1] - pad_size):
            # print(img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])
            new_img[i - pad_size][j - pad_size] += sum(sum(kernel * img[i - pad_size:i + pad_size + 1,
                                                                    j - pad_size:j + pad_size + 1]))

    return new_img


def get_gaussian_filter_1d(size, sigma):
    kernel = np.zeros(size, dtype=float)
    avg = size // 2  # 2
    for i in range(0, avg + 1):
        val = (1 / (2 * math.pi * sigma * sigma)) * math.exp((-1 * (i - avg) ** 2) / (2 * sigma ** 2))
        kernel[i] += val
        if i != avg:
            kernel[size - i - 1] += val

    kernel = kernel / sum(kernel)
    return kernel

    # print(np.dot(kernel.reshape(-1, 1), kernel.reshape(1, -1))) -> 2d


def get_gaussian_filter_2d(size, sigma):
    kernel = get_gaussian_filter_1d(size, sigma)
    kernel_2d = np.dot(kernel.reshape(-1, 1), kernel.reshape(1, -1))
    return kernel_2d