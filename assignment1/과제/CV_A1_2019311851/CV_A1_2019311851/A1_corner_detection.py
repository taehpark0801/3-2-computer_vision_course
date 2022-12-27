from A1_image_filtering import *
from A1_edge_detection import *
import cv2
import numpy as np


def compute_corner_response(img):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    # 3-2. a : sobel 필터 적용
    x_dir = cross_correlation_2d(img, sobel_x)
    y_dir = cross_correlation_2d(img, sobel_y)
    # 3-2. b :second moment M 정의
    second_moment = np.zeros((img.shape[0], img.shape[1], 4))

    xx = x_dir * x_dir
    xy = x_dir * y_dir
    yy = y_dir * y_dir
    # 3-2. b : second moment 계산
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            second_moment[i][j][0] += sum(sum(xx[i - 2:i + 3, j - 2:j + 3]))
            second_moment[i][j][1] += sum(sum(xy[i - 2:i + 3, j - 2:j + 3]))
            second_moment[i][j][2] += second_moment[i][j][1]
            second_moment[i][j][3] += sum(sum(yy[i - 2:i + 3, j - 2:j + 3]))

    # 3-2. c : r 나와 있는 대로 계산
    r = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r[i][j] += np.linalg.det(second_moment[i][j].reshape(2, 2)) - 0.04 * (
                        np.trace(second_moment[i][j].reshape(2, 2)) ** 2)

    # 3-2. d : 음수 0으로 나머지 normalize
    r[r < 0] = 0
    scaling_r = cv2.normalize(r, None, 0, 1, cv2.NORM_MINMAX)

    # df = pd.DataFrame(scaling_r)
    # df.to_csv('./scaling_r.csv', encoding='utf-8-sig')
    # cv2.imwrite('./scaling.png', scaling_r)
    # cv2.namedWindow('R')
    # scaling_img = scaling_r*255
    # cv2.imshow('R', scaling_img.astype(np.uint8))
    # cv2.waitKey(0)

    return scaling_r


def square_max(img, x, y, haf):
    m = np.max(img[x - haf: x + haf + 1, y - haf: y + haf + 1])
    for i in range(-haf, haf + 1):
        for j in range(-haf, haf + 1):
            if img[x + i][y + j] != m:
                img[x + i][y + j] = 0
    return img


def non_maximum_suppression_win(R, winSize=11):
    # R[R < 0.1] = 0
    haf = winSize // 2
    for i in range(0 + haf, R.shape[0] - haf):
        for j in range(0 + haf, R.shape[1] - haf):
            if R[i][j] != 0:
                R = square_max(R, i, j, haf)

    return R


if __name__ == "__main__":
    IMAGE_FILE_PATH = ['../lenna.png', '../shapes.png']  # 이미지 파일
    # IMAGE_FILE_PATH = ['../lenna.png']  # 이미지 파일
    for image in IMAGE_FILE_PATH:
        rd_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # print("Rd", rd_img.shape)
        # g_filter = get_gaussian_filter_2d(7, 1.5)
        # img = cross_correlation_2d(rd_img, g_filter)
        # 3-1.a & b
        g_filter = get_gaussian_filter_1d(7, 1.5)
        img = cross_correlation_1d(cross_correlation_1d(rd_img, g_filter.reshape(1, -1)), g_filter)

        # 3-2. e 시간 print
        start = time.time()
        corner = compute_corner_response(img)
        print("time :", time.time() - start)

        # 3-2. e 이미지 저장
        img_name = image.split("/")
        img_path = './result/part_3_corner_raw_' + img_name[-1]
        cv2.imwrite(img_path, corner*255)

        # 3-2. f 이미지 프린트
        cv2.namedWindow('corner_raw')
        corner255 = corner * 255
        cv2.imshow('corner_raw', corner255.astype(np.uint8))
        cv2.waitKey(0)

        # 3-3. a : 0.1보다 작은거 없애고 나머지는 초록색으로
        corner[corner < 0.1] = 0
        # color_corner = cv2.cvtColor(corner.astype(np.float32), cv2.COLOR_GRAY2BGR)
        # print(color_corner.shape)
        color_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
        # print(color_img.shape)

        # 초록색으로 프린트
        for i in range(color_img.shape[0]):
            for j in range(color_img.shape[1]):
                if corner[i][j] != 0:
                    color_img[i][j] = np.array([0, 255, 0])
        # 3-3. b : 이미지 저장
        img_path = './result/part_3_corner_bin_' + img_name[-1]
        cv2.imwrite(img_path, color_img)
        # 3-3. b :이미지 출력
        cv2.namedWindow('corner_bin')
        cv2.imshow('corner_bin', color_img.astype(np.uint8))
        cv2.waitKey(0)

        # 3-3.c
        suppression_win = non_maximum_suppression_win(corner, 11)
        color_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)

        """for i in range(color_img.shape[0]):
            for j in range(color_img.shape[1]):
                if suppression_win[i][j] != 0:
                    color_img[i][j] = np.array([0, 255, 0])"""
        # cv2.namedWindow('suppression_win')
        # cv2.imshow('suppression_win', color_img.astype(np.uint8))
        # cv2.waitKey(0)

        for i in range(color_img.shape[0]):
            for j in range(color_img.shape[1]):
                if suppression_win[i][j] != 0:
                    color_img = cv2.circle(color_img, (j, i), 4, (0, 255, 0), 2)

        cv2.namedWindow('suppression_circle')
        cv2.imshow('suppression_circle', color_img.astype(np.uint8))
        cv2.waitKey(0)

        # 3-3. d : 이미지 저장
        img_path = './result/part_3_corner_sup_' + img_name[-1]
        cv2.imwrite(img_path, color_img)