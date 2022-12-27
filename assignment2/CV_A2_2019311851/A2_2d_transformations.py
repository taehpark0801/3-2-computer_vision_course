import numpy as np
import math
import cv2
import time



def make_translation_matrix(key):
    matrix = np.eye(3, 3)

    if key == 97:
        matrix[1][2] = -5
    if key == 100:
        matrix[1][2] = 5
    if key == 119:
        matrix[0][2] = -5
    if key == 115:
        matrix[0][2] = 5
    return matrix


def make_rotate_matrix(key):
    matrix = np.eye(3, 3)
    r = np.radians(5)
    # r = np.pi / 36

    matrix[0][0] = np.cos(r)
    matrix[0][1] = -np.sin(r)
    matrix[1][0] = np.sin(r)
    matrix[1][1] = np.cos(r)

    # matrix[0][2] = 400 * (1 - np.cos(r)) + 400 * np.sin(r)
    # matrix[1][2] = 400 * (1 - np.cos(r)) - 400 * np.sin(r)

    if key == 116:
        matrix = np.linalg.inv(matrix)
    # print(matrix)
    return matrix


def make_flip_matrix(key):
    matrix = np.eye(3, 3)

    if key == 103:
        matrix[0][0] = -1
    if key == 102:
        matrix[1][1] = -1

    return matrix


def make_shearing_matrix(key):
    matrix = np.eye(3, 3)
    if key == 120:
        # matrix[0][0] -= 0.05
        matrix[0][0] = 0.95
    if key == 99:
        # matrix[0][0] += 0.05
        matrix[0][0] = 1.05
    if key == 121:
        # matrix[1][1] -= 0.05
        matrix[1][1] = 0.95
    if key == 117:
        # matrix[1][1] += 0.05
        matrix[1][1] = 1.05
    return matrix


def get_transformed_image(img, M):
    h, w = img.shape
    h = h // 2
    w = w // 2
    # print(start)
    # end = [400 + img.shape[0] // 2, 400 + img.shape[1] // 2]
    # print(end)
    # print(img.shape)
    # pad_h, pad_w = (801 - np.array(list(img.shape))) // 2
    # img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)
    plane = np.full((801, 801), 255, dtype=float)

    for i in range(-h, h + 1):
        for j in range(-w, w + 1):
            arr_tmp = M.dot(np.array([i, j, 1]).reshape(3,1)) + 400
            plane[int(arr_tmp[0][0])][int(arr_tmp[1][0])] = img[i + h][j + w]

    print(arr_tmp)
    # print(M)
    cv2.arrowedLine(plane, (400, 801), (400, 0), (0, 0, 0), 2, tipLength=0.01)
    cv2.arrowedLine(plane, (0, 400), (801, 400), (0, 0, 0), 2, tipLength=0.01)

    cv2.imshow('image', plane)


if __name__ == "__main__":
    IMAGE_FILE_PATH = ['../CV_Assignment_2_Images/smile.png']  # 이미지 파일
    for image in IMAGE_FILE_PATH:
        # print(image)
        rd_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # print(rd_img.shape)  # 101, 111
        # pad_h, pad_w = (801 - np.array(list(rd_img.shape))) // 2
        # print(pad_w, pad_h)  # 350, 345
        #
        # print(rd_img.shape)

        sum_matrix = np.eye(3, 3)
        scaling_matrix = np.eye(3, 3)
        rotation_matrix = np.eye(3, 3)
        translation_matrix = np.eye(3, 3)
        # sum_matrix[2][0] = 400
        # sum_matrix[2][1] = 400

        pad_h, pad_w = (801 - np.array(list(rd_img.shape))) // 2
        initial_img = np.pad(rd_img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)
        cv2.arrowedLine(initial_img, (401, 801), (401, 0), (0, 0, 0), 2, tipLength=0.01)
        cv2.arrowedLine(initial_img, (0, 401), (801, 401), (0, 0, 0), 2, tipLength=0.01)
        cv2.namedWindow('image')
        cv2.imshow('image', initial_img)
        # cv2.waitKey(0)
        img = rd_img
        while True:
            key = cv2.waitKeyEx()
            # key = 97
            # cv2.namedWindow('image')
            # cv2.imshow('image', rd_img)
            # cv2.waitKey(0)
            # print(key)

            # cv2.namedWindow('image')
            # cv2.imshow('image', rd_img)
            # cv2.waitKey(0)

            translation = [97, 100, 119, 115]
            rotate = [114, 116]
            flip = [102, 103]
            shearing = [120, 99, 121, 117]
            if key == 113:
                break
            if key in translation:
                # sum_matrix = np.dot(make_translation_matrix(key), sum_matrix)
                # sum_matrix = np.dot(sum_matrix, make_translation_matrix(key))
                sum_matrix = make_translation_matrix(key) @ sum_matrix
                # sum_matrix = translation_matrix @ rotation_matrix @ scaling_matrix
                # sum_matrix = sum_matrix @ scaling_matrix
                get_transformed_image(rd_img, sum_matrix @ scaling_matrix)
                # matrix = make_translation_matrix(key)
                # get_transformed_image(rd_img, matrix)
            if key in rotate:
                # sum_matrix = np.dot(sum_matrix, make_rotate_matrix(key))
                # rotation_matrix = make_rotate_matrix(key) @ rotation_matrix
                sum_matrix = make_rotate_matrix(key) @ sum_matrix
                get_transformed_image(rd_img, sum_matrix @ scaling_matrix)
                # matrix = make_rotate_matrix(key)
                # get_transformed_image(rd_img, matrix)
            if key in flip:
                # rotation_matrix = make_flip_matrix(key) @ rotation_matrix
                sum_matrix = make_flip_matrix(key) @ sum_matrix
                # sum_matrix = sum_matrix @ scaling_matrix
                # matrix_tmp = np.eye(3, 3)
                """if key == 103:
                    matrix_tmp[0][2] = 801
                if key == 102:
                    matrix_tmp[1][2] = 801"""
                # sum_matrix = matrix_tmp @ sum_matrix
                get_transformed_image(rd_img, sum_matrix @ scaling_matrix)
                # matrix = make_flip_matrix(key)
                # get_transformed_image(rd_img, matrix)
            if key in shearing:
                scaling_matrix = make_shearing_matrix(key) @ scaling_matrix
                # sum_matrix = make_shearing_matrix(key) @ sum_matrix
                # sum_matrix = np.dot(sum_matrix, make_shearing_matrix(key))
                # sum_matrix = sum_matrix @ scaling_matrix
                get_transformed_image(rd_img, sum_matrix @ scaling_matrix)
                # matrix = make_shearing_matri(key)
                # get_transformed_image(rd_img, matrix)
            if key == 104:
                sum_matrix = np.eye(3, 3)
                scaling_matrix = np.eye(3, 3)
                get_transformed_image(rd_img, sum_matrix)