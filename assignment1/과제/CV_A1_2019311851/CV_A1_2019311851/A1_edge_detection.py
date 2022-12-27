import pandas as pd

from A1_image_filtering import *
import cv2
import numpy as np
import time


def compute_image_gradient(img):
    # 2-2. a : 소벨 필터 적용
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    # print(img)

    # a. 소벨 필터 적용
    x_dir = cross_correlation_2d(img, sobel_x)
    y_dir = cross_correlation_2d(img, sobel_y)

    # cv2.namedWindow('x_dir')
    # cv2.imshow('x_dir', x_dir.astype('uint8'))
    # cv2.waitKey(0)

    # cv2.namedWindow('y_dir')
    # cv2.imshow('y_dir', y_dir.astype('uint8'))
    # cv2.waitKey(0)

    # b
    # print("x_dir**2", x_dir ** 2)
    # print("y_dir**2", y_dir ** 2)
    # 2-2. b : magnitude 구하기
    mag = np.sqrt(x_dir * x_dir + y_dir * y_dir)
    # print("mag", mag)

    # 2-2. b : direction 구하기
    dir = np.arctan2(y_dir, x_dir)
    # print("dir", dir)
    # cv2.namedWindow('mag')
    # cv2.imshow('mag', mag.astype(np.uint8))
    # cv2.waitKey(0)
    return mag, dir


def compare(i, j, x, y, mag, im, jm):
    if 0 <= i + x < im and 0 <= j + y < jm:
        if mag[i + x][j + y] >= mag[i][j]:
            mag[i][j] = 0

    return mag


def non_maximum_suppression_dir(mag, dir):
    # a
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360], dtype=float)
    # angles = np.array([0, 45, 90, 135, 180], dtype=np.float64)
    # dir = np.rad2deg(np.arctan(dir) * math.pi)
    # dir = np.rad2deg(dir)
    dir = dir * 180 / math.pi
    # print(dir)
    # df = pd.DataFrame(dir)
    # df.to_csv('./arctan_dir.csv', encoding='utf-8-sig')
    # print(dir.dtype)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if not np.isnan(dir[i][j]):
                if dir[i][j] < 0:
                    dir[i][j] += 360
                angle = angles[np.where(np.abs(angles - dir[i][j]) == min(np.abs(angles - dir[i][j])))][0]
                if angle == 360:
                    angle = 0
                dir[i][j] = angle
                # 죽이는거까지 한번에
                if angle >= 180:
                    angle -= 180
                if angle == 0:
                    mag = compare(i, j, 0, 1, mag, mag.shape[0], mag.shape[1])
                    mag = compare(i, j, 0, -1, mag, mag.shape[0], mag.shape[1])
                elif angle == 135:
                    mag = compare(i, j, -1, 1, mag, mag.shape[0], mag.shape[1])
                    mag = compare(i, j, 1, -1, mag, mag.shape[0], mag.shape[1])
                elif angle == 90:
                    mag = compare(i, j, -1, 0, mag, mag.shape[0], mag.shape[1])
                    mag = compare(i, j, 1, 0, mag, mag.shape[0], mag.shape[1])
                elif angle == 45:
                    mag = compare(i, j, -1, -1, mag, mag.shape[0], mag.shape[1])
                    mag = compare(i, j, 1, 1, mag, mag.shape[0], mag.shape[1])

    return mag


if __name__ == "__main__":
    IMAGE_FILE_PATH = ['../lenna.png', '../shapes.png']  # 이미지 파일
    # IMAGE_FILE_PATH = ['../shapes.png']  # 이미지 파일
    for image in IMAGE_FILE_PATH:
        rd_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        """cv2.namedWindow('magnitude')
        cv2.imshow('magnitude', rd_img)
        cv2.waitKey(0)"""
        # g_filter = get_gaussian_filter_2d(7, 1.5)
        # img = rd_img
        # img = cross_correlation_2d(rd_img, g_filter)
        # 2-1. a, b
        g_filter = get_gaussian_filter_1d(7, 1.5)
        img = cross_correlation_1d(cross_correlation_1d(rd_img, g_filter.reshape(1, -1)), g_filter)

        # cv2.namedWindow('filtered')
        # cv2.imshow('filtered', img.astype(np.uint8))
        # cv2.waitKey(0)

        # 2-2. d 계산 속도 출력
        start = time.time()
        magnitude, direction = compute_image_gradient(img)
        print("cig_time :", time.time() - start)

        # 2-2. d : 이미지 저장
        img_name = image.split("/")
        img_path = './result/part_2_edge_raw_' + img_name[-1]
        cv2.imwrite(img_path, magnitude)

        # 2-2. d 그림 출력
        cv2.namedWindow('edge_raw')
        cv2.imshow('edge_raw', cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        cv2.waitKey(0)



        # 2-3.
        start = time.time()
        sup_mag = non_maximum_suppression_dir(magnitude, direction)
        print("sup_time :", time.time() - start)

        # 2-2. d : 이미지 저장
        img_path = './result/part_2_edge_sup_' + img_name[-1]
        cv2.imwrite(img_path, magnitude)

        cv2.namedWindow('sup_mag')
        cv2.imshow('sup_mag', cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        cv2.waitKey(0)


