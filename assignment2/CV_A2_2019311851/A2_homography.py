import cv2
import numpy as np
import random
import time
from imgae_filtering import *


def normalize_and_scaling_matrix(srcP):
    # 2-2. b - normalize
    # transformations as two 3 × 3matrices
    # translation scaling matrix 활용해도..?? good 근데 결과가...??
    mx, my = np.mean(srcP, axis=0)
    mean_matrix = np.array([[1, 0, -mx], [0, 1, -my], [0, 0, 1]])
    mean_srcp = srcP - np.array([mx, my])

    distance = np.hypot(mean_srcp[:, 0], mean_srcp[:, 1])
    """distance = []
        for p in mean_srcp:
            distance.append(p[0] ** 2 + p[1] ** 2)
        max_d = max(distance)
        scale_num = np.sqrt(2) / max_d"""

    max_d = np.max(distance)
    scale_num = np.sqrt(2) / max_d

    # scaling_matrix = np.array([[scale_num * 2, 0, 0], [0, scale_num * 2, 0], [0, 0, 1]])
    scaling_matrix = np.array([[scale_num, 0, 0], [0, scale_num, 0], [0, 0, 1]])
    # tmp_matrix = np.sqrt(np.power(mean_srcp, 2) * np.sqrt(2))

    # srcp_matrix = []
    # for p in mean_srcp:
    #     srcp_matrix.append(np.power(list(p) + [1], 2).reshape(3, 1) * np.sqrt(2))

    # result = []
    # for p in srcp_matrix:
    #     result.append(scaling_matrix @ tmp_matrix)

    """distance1 = []
    for p in result:
        distance1.append(p[0] ** 2 + p[1] ** 2)
    print(np.sqrt(2))
    print(distance1)
    print(max(distance1))"""
    result = np.dot(scaling_matrix, mean_matrix)

    return np.array(result)


# 아래는 normalize_and_scaling은 matrix 사용 안해서 폐기함
# 근데 위에꺼 왜 결과 안나옴;; 이걸로 하고 위에꺼는 나중에 생각하쟈

""" print(srcP)
mx, my = np.mean(srcP, axis=0)
print(mx, my)
mean_srcp = []
for p in srcP:
    # print(np.array(list(p)))
    mean_srcp.append(np.array(list(p)) - np.array([mx, my]))
print(np.mean(mean_srcp, axis=0))

# scaling
distance = []
for p in srcP:
    distance.append(p[0] ** 2 + p[1] ** 2)
max_d = max(distance)
# scale_num = np.sqrt(2) / max_d

scale_srcp = []
for p in srcP:
    tmp = []
    pn0, pn1 = int(p[0] / np.abs(p[0])), int(p[1] / np.abs(p[1]))
    tmp.append(np.sqrt((p[0] ** 2 * np.sqrt(2)) / max_d) * pn0)
    tmp.append(np.sqrt((p[1] ** 2 * np.sqrt(2)) / max_d) * pn1)
    scale_srcp.append(tmp)

return scale_srcp"""


def normalize_and_scaling_compute(srcp, Ts):
    # 2-2. Td, Ts
    # comput normalized point x, y
    srcp_matrix = np.array(srcp)
    srcp_matrix = np.pad(srcp_matrix, ((0, 0), (0, 1)), 'constant', constant_values=1)

    # print(srcp_matrix)
    scaled_srcp = []
    for m in srcp_matrix:
        scaled_srcp.append(np.dot(Ts, m.reshape(3, 1)))
    # print(np.array(scaled_srcp).reshape(-1, 3))
    # print(np.array(scaled_srcp).reshape(-1, 3).shape)
    return np.array(scaled_srcp).reshape(-1, 3)


def compute_homography(srcP, destP):
    # N = 4
    # srcP = srcP[:N]
    # destP = destP[:N]
    # print(srcP, destP)
    # srcP = normalize_and_scaling_compute(srcP)
    # destP = normalize_and_scaling_compute(destP)

    """distance1 = []
    for p in srcP:
        distance1.append(p[0] ** 2 + p[1] ** 2)
    print(np.sqrt(2))
    print(max(distance1))

    distance1 = []
    for p in destP:
        distance1.append(p[0] ** 2 + p[1] ** 2)
    print(np.sqrt(2))
    print(max(distance1))

    print(np.mean(srcP, axis=0), np.mean(destP, axis=0))"""

    A = []
    # print("srcp", srcP)
    for i in range(len(srcP)):
        x = srcP[i][0]
        y = srcP[i][1]
        u = destP[i][0]
        v = destP[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)
    # print(A)
    u, s, v = np.linalg.svd(A, full_matrices=True)
    # print("u", u)
    # print("s", s)
    # print("v", v)
    H = v[-1].reshape(3, 3)

    # H = H / H[2][2]
    return H


def compute_homography_ransac(srcP, destP, th):
    start = time.time()
    # print(srcP)
    rand_num_lst = [i for i in range(len(srcP))]
    min_th = th
    min_cnt = 4
    err_min = np.inf
    final_H = 0

    while time.time() - start < 4:
        rand_num = random.sample(rand_num_lst, 15)
        random_srcP = np.array([srcP[i] for i in rand_num])
        random_destP = np.array([destP[i] for i in rand_num])
        """A = []
        for i in range(len(random_srcP)):
            x = random_srcP[i][0]
            y = random_srcP[i][1]
            u = random_destP[i][0]
            v = random_destP[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A = np.array(A)
        # print(A)
        u, s, v = np.linalg.svd(A, full_matrices=True)
        # print("u", u)
        # print("s", s)
        # print("v", v)
        H = v[-1].reshape(3, 3)"""
        # print("random",  random_srcP)
        # print(len(random_srcP))
        H = compute_homography(random_srcP, random_destP)
        estimate_desp = np.dot(random_srcP, H)
        err_lst = np.hypot(random_srcP[:, 0] - estimate_desp[:, 0], random_srcP[:, 1] - estimate_desp[:, 1])
        # print(err_lst)
        # print(random_srcP)
        err_th = []

        for i in range(len(err_lst)):
            if err_lst[i] < th:
                err_th.append(i)
        # print(err_th)
        # print(random_srcP[0])
        """if len(err_th) >= min_cnt:
            if len(err_th) == min_cnt and err_min < np.sum(err_lst):
                continue
            min_cnt = len(err_th)
            err_min = np.sum(err_lst)
            new_srcp = np.array([random_srcP[i] for i in err_th])
            new_desp = np.array([random_destP[i] for i in err_th])
            # print("new", new_srcp)
            final_H = compute_homography(new_srcp, new_desp)"""

        # distance = np.hypot(mean_srcp[:, 0], mean_srcp[:, 1])
        # print(random_srcP)
        # print(np.hypot(random_srcP[:, 0], random_srcP[:, 1]))

        # err_lst = np.hypot(random_srcP[:, 0], random_srcP[:, 1]) - np.hypot(estimate_desp[:, 0], estimate_desp[:, 1])
        # print(err_lst)
        err = np.abs(np.sum(np.hypot(random_srcP[:, 0] - estimate_desp[:, 0], random_srcP[:, 1] - estimate_desp[:, 1])))
        # print(err)
        # th_lst = [i for i in range(len(err_lst)) if err_lst[i] < th]
        # print(th_lst)
        # final_H = compute_homography(random_srcP, random_destP)

        if min_th > err:
            min_th = err
            final_H = H

    return final_H


def compute_Dmatch(des1, des2, th):
    similarity = []
    for i in range(len(des1)):
        tmp = []
        for j in range(len(des2)):
            hamming_distance = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
            tmp.append([i, j, hamming_distance])
        tmp.sort(key=lambda x: x[2])
        for k in range(len(tmp)):
            tmp[k].append(tmp[0][2] / tmp[1][2])  # threshold
        # print(tmp)
        similarity.append(tmp)
    # print(similarity)
    # print(similarity)
    # similarity = np.array(similarity)
    # print(np.array(similarity).shape)
    # print(similarity[:, :1, 2])
    # print(np.array(similarity))
    if th == True:
        min_hamming = np.min(np.array(similarity)[:, :1, 2])
    # print(min_hamming)
        for i in range(len(similarity) - 1, -1, -1):
            if similarity[i][0][2] > min_hamming * 3:
                del similarity[i]

    similarity.sort(key=lambda x: x[0][2])
    # similarity sort(similarity, key=lambda x: x[0][2])
    # print(similarity[:10])
    sim_match = []

    if th == False:
        for i in similarity:
            if i[0][2] / i[1][2] <= 0.8:
                sim_match.append(i[0])
    else:
        for i in similarity:
            if i[0][2] / i[1][2] <= 0.8:
                sim_match.append(i[0])
    # print(sim_match)
    # sim_match.sort(key=lambda x: x[3] ** 2 * x[2])
    if th == True:
        sim_match.sort(key=lambda x: x[3] ** 2 * x[2])
        # sim_match.sort(key=lambda x: x[3] * 2 * x[2] ** 2)
    # sim_match.sort(key=lambda x: x[3])
    # print(len(sim_match))
    dmatches = []
    for i in range(len(sim_match)):
        # Dmatch(queryIdx, trainIdx, distance)
        # queryIdx : 1번 이미지의 특징점 번호
        # trainIdx : 2번 이미지의 특징점 번호
        dmatch = cv2.DMatch(sim_match[i][0], sim_match[i][1], sim_match[i][2])
        # dmatch.create( )
        dmatches.append(dmatch)

    return sim_match, dmatches


def wrap(img1, img2):
    h, w = img1.shape
    for i in range(h):
        for j in range(w):
            if img1[i][j] != 0:
                img2[i][j] = img1[i][j]

    cv2.imshow('dst', img2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def gradient(img, h, w):
    kernel = get_gaussian_filter_2d(15, 7)
    pad_size = kernel.shape[0] // 2
    img_h = img.shape[0]
    img_w = img.shape[1]

    # 패딩
    for i in range(pad_size):
        img = np.insert(img, img_h + i * 2, img[-1], axis=0)
        img = np.insert(img, 0, img[0], axis=0)
        img = np.insert(img, img_w + i * 2, img.T[-1], axis=1)
        img = np.insert(img, 0, img.T[0], axis=1)

    new_img = np.zeros((img_h, img_w), dtype=float)
    for i in range(0 + pad_size, img.shape[0] - pad_size):
        for j in range(0 + pad_size, img.shape[1] - pad_size):
            # print(img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])
            if w - 20 < j <= w + 20:
                new_img[i - pad_size][j - pad_size] += sum(sum(kernel * img[i - pad_size:i + pad_size + 1,
                                                                           j - pad_size:j + pad_size + 1]))
            else:
                new_img[i - pad_size][j - pad_size] += img[i - pad_size][j - pad_size]
    print(new_img)
    cv2.imshow('dst', new_img.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 2-1
    rd_img1 = cv2.imread('../CV_Assignment_2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    rd_img2 = cv2.imread('../CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)

    # kp = 좌표(pt), 지름(size), 각도(angle), 응답(response), 옥타브(octave), 클래스 ID(class_id)
    # des = 각 kp을 설명하기 위한 2차원 배열로 표현. 배열은 두 kp이 같은지 판단할 때 사용
    # a, b read pair of images & use ORB
    orb1 = cv2.ORB_create()
    kp1 = orb1.detect(rd_img1, None)
    kp1, des1 = orb1.compute(rd_img1, kp1)
    # print(des1)

    orb2 = cv2.ORB_create()
    kp2 = orb2.detect(rd_img2, None)
    kp2, des2 = orb2.compute(rd_img2, kp2)

    # c. Perform feature matching between two images
    fsim_match, fdmatches = compute_Dmatch(des2, des1, th=False)

    # d. drawMatches
    dst = cv2.drawMatches(rd_img2, kp2, rd_img1, kp1, tuple(fdmatches[:10]), None, flags=2)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 2-2. Computing homography with normalization
    sim_match, dmatches = compute_Dmatch(des1, des2, th=True)

    N = 15
    srcp = []
    destp = []
    # srcp = np.float32([kp1[m.queryIdx].pt for m in dmatches[:N]])
    # destp = np.float32([kp2[m.trainIdx].pt for m in dmatches[:N]])

    for i in range(len(dmatches)):
        srcp.append(list(kp1[dmatches[i].queryIdx].pt))
        destp.append(list(kp2[dmatches[i].trainIdx].pt))
    # srcp, destp = np.array(srcp).reshape((-1, 1, 2)), np.array(destp).reshape((-1, 1, 2))
    srcp, destp = np.array(srcp), np.array(destp)
    print(srcp)

    Ts, Td = normalize_and_scaling_matrix(srcp), normalize_and_scaling_matrix(destp)
    norm_srcp, norm_destp = normalize_and_scaling_compute(srcp, Ts), normalize_and_scaling_compute(destp, Td)
    # print(norm_srcp)
    # print("srcp", srcp)
    # print("srcp_norm", norm_srcp)

    H = compute_homography(norm_srcp[:N], norm_destp[:N])
    # H = compute_homography(srcp, destp)
    # H = compute_homography(destp, srcp)
    # print(H)
    # H = np.dot(np.dot(np.linalg.inv(Ts), H), Td)
    # print(H)
    # print(rd_img1, rd_img1.shape)
    # src = np.float32([kp1[m.queryIdx].pt for m in dmatches[:15]]).reshape((-1, 1, 2))
    # dst = np.float32([kp2[m.trainIdx].pt for m in dmatches[:15]]).reshape((-1, 1, 2))
    # H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    # H, status = cv2.findHomography(srcp, destp, cv2.RANSAC, 5.0)
    # H, status = cv2.findHomography(norm_srcp, norm_destp, cv2.RANSAC, 5.0)
    H = np.dot(np.dot(np.linalg.inv(Td), H), Ts)
    # H = np.linalg.inv(Td) @ H @ Ts
    im_dst = cv2.warpPerspective(rd_img1, H, (rd_img2.shape[1], rd_img2.shape[0]))
    cv2.imshow('dst', im_dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 2-4. image warping
    wrap(im_dst, rd_img2)

    # 2-3. Computing homography with RANSAC
    rd_img2 = cv2.imread('../CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
    H = compute_homography_ransac(norm_srcp[:N + 5], norm_destp[:N + 5], 10)
    H = np.dot(np.dot(np.linalg.inv(Td), H), Ts)
    im_dst = cv2.warpPerspective(rd_img1, H, (rd_img2.shape[1], rd_img2.shape[0]))
    cv2.imshow('dst', im_dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 2-4. image warping

    wrap(im_dst, rd_img2)

    rd_img3 = cv2.imread('../CV_Assignment_2_Images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    print(rd_img1.shape)
    print(rd_img3.shape)
    rd_img3 = cv2.resize(rd_img3, (rd_img1.shape[1], rd_img1.shape[0]), interpolation=cv2.INTER_LINEAR)
    im_dst = cv2.warpPerspective(rd_img3, H, (rd_img2.shape[1], rd_img2.shape[0]))

    cv2.imshow('dst', im_dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    wrap(im_dst, rd_img2)

    # 2-5. image stitching
    rd_img1 = cv2.imread('../CV_Assignment_2_Images/diamondhead-11.png', cv2.IMREAD_GRAYSCALE)
    rd_img2 = cv2.imread('../CV_Assignment_2_Images/diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
    # print(rd_img1.shape)

    orb1 = cv2.ORB_create()
    kp1 = orb1.detect(rd_img1, None)
    kp1, des1 = orb1.compute(rd_img1, kp1)
    # print(des1)

    orb2 = cv2.ORB_create()
    kp2 = orb2.detect(rd_img2, None)
    kp2, des2 = orb2.compute(rd_img2, kp2)

    sim_match, dmatches = compute_Dmatch(des1, des2, th=True)
    N = 15
    srcp = []
    destp = []
    # srcp = np.float32([kp1[m.queryIdx].pt for m in dmatches[:N]])
    # destp = np.float32([kp2[m.trainIdx].pt for m in dmatches[:N]])

    for i in range(len(dmatches)):
        srcp.append(list(kp1[dmatches[i].queryIdx].pt))
        destp.append(list(kp2[dmatches[i].trainIdx].pt))
    srcp, destp = np.array(srcp), np.array(destp)

    Ts, Td = normalize_and_scaling_matrix(srcp), normalize_and_scaling_matrix(destp)
    norm_srcp, norm_destp = normalize_and_scaling_compute(srcp, Ts), normalize_and_scaling_compute(destp, Td)

    H = compute_homography_ransac(norm_srcp[:N + 5], norm_destp[:N + 5], 10)
    H = np.dot(np.dot(np.linalg.inv(Td), H), Ts)
    im_dst = cv2.warpPerspective(rd_img1, H, ((rd_img1.shape[1] + rd_img2.shape[1]), rd_img2.shape[0]))
    im_dst[0:rd_img2.shape[0], 0:rd_img2.shape[1]] = rd_img2
    # im_dst[0:rd_img1.shape[0], 0:rd_img1.shape[1]] = rd_img1

    cv2.imshow('dst', im_dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # gradient(im_dst, rd_img2.shape[0], rd_img2.shape[1])







