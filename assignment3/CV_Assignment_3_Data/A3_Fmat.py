import random
import numpy as np
import cv2
import time
from compute_avg_reproj_error import *


def compute_F_raw(m):
    # print(len(m))
    # rand_num = [i for i in range(len(m))]
    # rand_num = random.sample(rand_num_lst, len(m))
    A = []
    for i, mat in enumerate(m):
        x1 = mat[0]
        y1 = mat[1]
        x2 = mat[2]
        y2 = mat[3]
        # A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        A.append([x1 * x2, x2 * y1, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])
    A = np.array(A)
    u, s, v = np.linalg.svd(A)
    F = v[-1].reshape(3, 3)
    return F


def norm_matrix(m, img):
    h, w, _ = img.shape
    mean_matrix = np.array([[1, 0, -w // 2], [0, 1, -h // 2], [0, 0, 1]])

    scaling_matrix = np.array([[1 / (w // 2), 0, 0], [0, 1 / (h // 2), 0], [0, 0, 1]])
    return np.dot(scaling_matrix, mean_matrix)


def normalization(m, img):
    # 과제 2번의 normalize_and_scaling_matrix 활용
    w, h, _ = img.shape
    # scaling_matrix = np.array([[1 / (w // 2), 0, 0], [0, 1 / (h // 2), 0], [0, 0, 1]])
    Ts = norm_matrix(m, img)

    # 과제 2번의 normalize_and_scaling_compute 활용
    m_matrix = np.array(m)
    m_matrix = np.pad(m_matrix, ((0, 0), (0, 1)), 'constant', constant_values=1)

    scaled_m = []
    for m in m_matrix:
        scaled_m.append(np.dot(Ts, m.reshape(3, 1)))
    return np.array(scaled_m)[:, :2].reshape(-1, 2)


def compute_F_norm(M, img):
    # normalization
    m = M.copy()
    m[:, :2] = normalization(m[:, :2], img)
    m[:, 2:] = normalization(m[:, 2:], img)
    # print(m)
    F = compute_F_raw(m)

    # constraint rank 2
    u, s, v = np.linalg.svd(F)
    s[2] = 0
    F = np.dot(u, np.dot(np.diag(s), v))

    # un - normalize
    Ts = norm_matrix(m[:, :2], img)  # img1 나 img2 Ts는 같음
    F = np.dot(np.dot(Ts.T, F), Ts)
    return F


def compute_F_mine(m, img):
    F = compute_F_norm(m, img)
    err = compute_avg_reproj_error(m, F)
    start = time.time()
    while time.time() - start < 3:
        randoms = random.sample([i for i in range(len(m))], 60)
        tmp_m = np.array([m[i] for i in randoms])
        # print(tmp_m)
        tmp_F = compute_F_norm(tmp_m, img)
        tmp_err = compute_avg_reproj_error(tmp_m, tmp_F)
        if err > tmp_err:
            F = tmp_F
            err = tmp_err
    return F


def epipolarline(P, F):
    ep_line = []
    for i, p in enumerate(P):
        x1 = p[0]
        y1 = p[1]
        l1 = np.dot(F, np.array([x1, y1, 1]).reshape(3, 1))
        # l2 = np.dot(F.T, np.array([x2, y2, 1]).reshape(3, 1))
        ep_line.append(l1)
        # ep_line.append(l2)
    return np.array(ep_line)


def draw_ep_lines(m, image1, image2, F1, F2):
    M = m.copy()
    randoms = random.sample([i for i in range(len(M))], 3)
    # point = np.array([m[i] for i in randoms])
    point1 = np.array([m[i][:2] for i in randoms])
    point2 = np.array([m[i][2:] for i in randoms])
    #print(epipolarline(point, F))

    img1 = image1.copy()
    img2 = image2.copy()
    r, c, _ = img1.shape
    # F2 = [F1[3], F1[2], F1[1], F1[0]]
    ep = epipolarline(point2, F2).reshape(-1, 3)
    # print(ep)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    for r, pt1, pt2 in zip(ep, point1, point2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        # print((x0, y0), (x1, y1))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        pt1, pt2 = map(int, pt1), map(int, pt2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        # img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    ep = epipolarline(point1, F1).reshape(-1, 3)
    for r, pt1, pt2 in zip(ep, point1, point2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        pt1, pt2 = map(int, pt1), map(int, pt2)
        # img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    # cv2.imshow('img1', img1)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imshow('img2', img2)
    return img1, img2



if __name__ == '__main__':
    #temple image
    temple_M = np.loadtxt('temple_matches.txt')
    temple_img1 = cv2.imread('temple1.png')
    temple_img2 = cv2.imread('temple2.png')

    print("Average Reprojection Error (temple1.png and temple2.png)")
    temple_f1 = compute_F_raw(temple_M)
    print("Raw = ", compute_avg_reproj_error(temple_M, temple_f1))
    temple_f2 = compute_F_norm(temple_M, temple_img1)
    tmp = np.zeros(temple_M.shape)
    tmp[:, :2] = temple_M[:, 2:]
    tmp[:, 2:] = temple_M[:, :2]
    temple_f22 = compute_F_norm(tmp, temple_img2)
    print("Norm = ", compute_avg_reproj_error(temple_M, temple_f2))

    f3 = compute_F_mine(temple_M, temple_img1)
    print("Mine = ", compute_avg_reproj_error(temple_M, f3))

    # point = np.array([M[i] for i in randoms])

        # cv2.waitKey()

    # ep = epipolarline(point, f2)
    # print(ep)

    # house image
    house_M = np.loadtxt('house_matches.txt')
    house_img1 = cv2.imread('house1.jpg')
    house_img2 = cv2.imread('house2.jpg')

    print("Average Reprojection Error (house1.jpg and house2.jpg)")
    house_f1 = compute_F_raw(house_M)
    # print(f1)
    print("Raw = ", compute_avg_reproj_error(house_M, house_f1))
    house_f2 = compute_F_norm(house_M, house_img1)
    tmp = np.zeros(house_M.shape)
    tmp[:, :2] = house_M[:, 2:]
    tmp[:, 2:] = house_M[:, :2]
    house_f22 = compute_F_norm(tmp, house_img1)
    print("Norm = ", compute_avg_reproj_error(house_M, house_f2))
    f3 = compute_F_mine(house_M, house_img1)
    print("Mine = ", compute_avg_reproj_error(house_M, f3))



    # library image
    library_M = np.loadtxt('library_matches.txt')
    library_img1 = cv2.imread('library1.jpg')
    library_img2 = cv2.imread('library2.jpg')

    print("Average Reprojection Error (library1.jpg and library2.jpg)")
    library_f1 = compute_F_raw(library_M)
    print("Raw = ", compute_avg_reproj_error(library_M, library_f1))
    library_f2 = compute_F_norm(library_M, library_img1)
    tmp = np.zeros(library_M.shape)
    tmp[:, :2] = library_M[:, 2:]
    tmp[:, 2:] = library_M[:, :2]
    library_f22 = compute_F_norm(tmp, library_img2)
    print("Norm = ", compute_avg_reproj_error(library_M, library_f2))
    f3 = compute_F_mine(library_M, library_img1)
    print("Mine = ", compute_avg_reproj_error(library_M, f3))


    while True:
        key = cv2.waitKeyEx()
        if key == 113:
            cv2.destroyAllWindows()
            break

        ep_img1, ep_img2 = draw_ep_lines(temple_M, temple_img1, temple_img2, temple_f2, temple_f22)
        # ep_img2 = draw_ep_lines(M, rd_img2, rd_img1, f2)
        cv2.imshow('img1', cv2.hconcat([ep_img1, ep_img2]))

    while True:
        key = cv2.waitKeyEx()
        if key == 113:
            cv2.destroyAllWindows()
            break

        ep_img1, ep_img2 = draw_ep_lines(house_M, house_img1, house_img2, house_f2, house_f22)
        # ep_img2 = draw_ep_lines(M, rd_img2, rd_img1, f2)
        cv2.imshow('img1', cv2.hconcat([ep_img1, ep_img2]))

    while True:
        key = cv2.waitKeyEx()
        if key == 113:
            cv2.destroyAllWindows()
            break

        ep_img1, ep_img2 = draw_ep_lines(library_M, library_img1, library_img2, library_f2, library_f22)
        # ep_img2 = draw_ep_lines(M, rd_img2, rd_img1, f2)
        cv2.imshow('img1', cv2.hconcat([ep_img1, ep_img2]))

