import numpy as np
import os
from sklearn.cluster import KMeans


def kmeans(sifts):
    model = KMeans(n_clusters=128, random_state=10, max_iter=1000)

    # 학습 코드
    # model.fit(sifts)
    # print(model.cluster_centers_)
    # np.save('cluster5', model.cluster_centers_)
    # np.save('predict', model.predict(sifts))

    # 사전학습 load
    centers = np.load('cluster5.npy')
    # predict = np.load('predict.npy')
    return centers


def histo(sifts, centers):
    hist_arr = []
    center_n = len(centers)
    for sift in sifts:
        l2_norm = [0 for _ in range(center_n)]
        for feature in sift:
            tmp = np.zeros(len(centers))
            for i, center in enumerate(centers):
                tmp[i] += np.linalg.norm(feature - center, ord=2)
            n = np.argmin(tmp)
            l2_norm[n] += 1
            number = tmp[n]
            tmp[n] = np.inf
            n = np.argmin(tmp)
            if number * 1.5 > tmp[n]:
                l2_norm[n] += 1
        hist_arr.append(l2_norm)
    return np.array(hist_arr, dtype="float32")


if __name__ == '__main__':
    file_names = os.listdir('../CV_A4_Feats/feats/')
    print(len(file_names))
    sift_array = []
    sift_array2 = []

    # f = open("CV_A4_Feats/feats/00000.sift", "r")
    # a = np.fromfile(f, dtype=np.ubyte)
    # print(len(a), a)
    for f in file_names:
        fname = './feats/' + f
        # print(np.fromfile(fname, dtype=float).shape) #다 길이가 다름
        # sift_array = np.append(sift_array, np.fromfile(fname, dtype=np.ubyte))
        read = np.fromfile(fname, dtype=np.ubyte)
        sift_array.append(list(read))
        sift_array2.append(read.reshape(-1, 128))
    sift_array = np.concatenate(sift_array)
    sift_array = sift_array.reshape(-1, 128)
    # print(sift_array, sift_array.shape)
    k_center = kmeans(sift_array)
    # print(len(k_center))
    bovw = histo(sift_array2, k_center)
    # print(bovw)

    with open('A4_2019311851.des', 'wb') as fp:
        nd = np.array([1000, len(k_center)], dtype=int)
        fp.write(nd.tobytes())
        fp.write(bovw.tobytes())
