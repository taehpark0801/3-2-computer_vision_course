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
    # return kernel_2d / sum(sum(kernel_2d))


if __name__ == "__main__":
    IMAGE_FILE_PATH = ['../lenna.png', '../shapes.png']  # 이미지 파일
    # d_kernel = np.array([-1, 0, 1])
    # d_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # d_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # d_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    print(get_gaussian_filter_1d(5, 1))
    print(get_gaussian_filter_2d(5, 1))
    for image in IMAGE_FILE_PATH:
        rd_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        ker = [5, 11, 17]
        sig = [1, 6, 11]
        images = []

        for i in ker:
            tmp = []
            for j in sig:
                kernel = get_gaussian_filter_1d(i, j)
                tmp_img = cross_correlation_1d(cross_correlation_1d(rd_img, kernel.reshape(1, -1)), kernel)
                text = str(i) + "x" + str(i) + " s=" + str(j)
                tmp_img = cv2.putText(tmp_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                tmp.append(tmp_img)
            images.append(np.hstack(tmp))


        conc_image = np.vstack(images)

        # 이미지 저장
        img_name = image.split("/")
        img_path = './result/part_1_gaussian_filtered_' + img_name[-1]
        cv2.imwrite(img_path, conc_image)

        cv2.namedWindow('image')
        cv2.imshow('image', cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        cv2.waitKey(0)

        #1-2. d: 1d와 2d 비교
        #1D
        kernel = get_gaussian_filter_1d(11, 11)
        before_cross_sum = sum(sum(rd_img))
        start = time.time()
        tmp_img_1d = cross_correlation_1d(cross_correlation_1d(rd_img, kernel.reshape(1, -1)), kernel)
        print("1D computation time :", time.time() - start)

        #2D
        kernel = get_gaussian_filter_2d(11, 11)
        start = time.time()
        tmp_img_2d = cross_correlation_2d(rd_img, kernel)
        print("2D computation time :", time.time() - start)

        #sum of (absolute) intensity differences
        pixel_wise_dif = tmp_img_1d - tmp_img_2d
        cv2.namedWindow('pixel_wise_dif image')
        cv2.imshow('pixel_wise_dif image', pixel_wise_dif)
        cv2.waitKey(0)

        print("pixel_wise_dif", np.sum(np.sum(np.abs(pixel_wise_dif))))





    # arr2 = arr2.astype(np.uint16)
    # print(arr2.dtype)

    # print(np.version)
    # rd_img = cv2.imread('../lenna.png', cv2.IMREAD_GRAYSCALE)
    # cross_correlation_2d(rd_img, get_gaussian_filter_2d(7, 1.5))
    # print(sum(sum(get_gaussian_filter_2d(7, 1.5))))
    # print(genGaussianKernel(7, 1.5))
