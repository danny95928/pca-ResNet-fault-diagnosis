import numpy as np
import scipy.io as scio
import os
import pickle
from sklearn.decomposition import PCA
from PIL import Image
from collections import Counter
import pandas as pd
from typing import Optional


# 均值、标准差、均方根、峰峰值、偏度、峰度、波形因子、峰值因子、裕度因子、脉冲因子
def feature_extraction(
        dataset: Optional[np.ndarray]
):
    pending_column = []

    for data in dataset:
        mean, var = data.mean(), data.var()
        rms = np.sqrt(
                        pow(data.mean(), 2)
                    +
                        pow(data.std(),  2)
                     )
        peak_value = max(data) - min(data)
        skewness = pd.Series(data).skew()
        kurtosis = (
                    np.sum(
                               [x ** 4 for x in data]
                    ) /
                    len(data)
                   ) / pow(rms, 4)
        sum = 0
        for p1 in range(len(data)):
            sum += np.sqrt(abs(data[p1]))

        shape_factor = rms / (abs(data).mean())
        crest_factor = (max(data)) / rms
        impulse_factor = (max(data)) / (abs(data).mean())
        margin_factor = max(data) / pow(sum / (len(data)), 2)

        feature_list = [mean, var, peak_value, skewness, kurtosis, shape_factor,
                        crest_factor, impulse_factor, margin_factor]

        pending_column.append(feature_list)
    return np.array(pending_column)


def random_choice_id():
    ids = ''
    alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for _ in range(25):
        mo = np.random.choice(alphabets)
        ids += mo
    return ids


def save_pickle(data, pickle_dir):
    file = open(pickle_dir, 'wb')
    pickle.dump(data, file)
    file.close()


def load_mat_return_array(path, label, time_step=100, seq_len=1024):
    keys = []
    data = scio.loadmat(path)
    for key in data.keys():
        keys.append(key)
    X = data[keys[3]]
    idx = []
    for i, _ in enumerate(X):
        if i % time_step == 0:
            idx.append(i)
    idx = idx[:-11]
    data = []
    for id in idx:
        data.append(
            X[id: id + seq_len].
            reshape(seq_len).tolist() + [label]
        )
    return np.array(data)


def make_train_set(dataset_dir):
    namelist = os.listdir(dataset_dir)

    c = np.zeros(shape=(1, 1025))
    for name, label in zip(namelist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        mat_path = os.path.join(dataset_dir, name)

        data_only = load_mat_return_array(
            path=mat_path,
            label=label
        )
        c = np.vstack((c, data_only))
    c = c[1:]

    np.random.seed(1)
    np.random.shuffle(c)

    train_set, label = c[:, :-1], c[:, -1:]

    train_set = (
                        train_set - np.min(train_set)
                ) / (
                        np.max(train_set) - np.min(train_set)
                )

    dataset = np.hstack(tup=(train_set, label))

    pca = PCA(n_components=256)
    pca.fit(train_set)
    pca_train_set = pca.transform(train_set)

    pca_train_set = (
                        pca_train_set - np.min(pca_train_set)
                ) / (
                        np.max(pca_train_set) - np.min(pca_train_set)
                )

    features = feature_extraction(pca_train_set)
    print(features.shape)
    pca_add_features = np.hstack(tup=(pca_train_set, features))

    train_2_v = int(len(dataset) * 0.8)

    pca_train, feature,  pca_add_feature = {}, {}, {}
    gru_train, gru_test = {}, {}
    train_count, test_count = [], []

    for position, one_data, pca_one_data, feature_one, pca_add_one in zip(
            range(1, len(dataset) + 1),
            dataset,
            pca_train_set,
            features,
            pca_add_features):

        key = random_choice_id()

        if train_2_v >= position:
            gru_train[key] = one_data
            train_count.append(int(one_data[-1]))

        else:
            gru_test[key] = one_data
            test_count.append(int(one_data[-1]))

        pca_train[key] = pca_one_data
        feature[key] = feature_one
        pca_add_feature[key] = pca_add_one

        im = Image.fromarray(
            np.uint8(
                one_data[:-1].reshape(32, 32) * 255.
            )
        )
        im = im.convert('L')
        im.save(r'IMAGE/{}.png'.format(key))

    save_pickle(data=gru_train, pickle_dir='pickle_file/train.p')
    save_pickle(data=gru_test, pickle_dir='pickle_file/test.p')
    save_pickle(data=pca_train, pickle_dir='pickle_file/pca_train.p')
    save_pickle(data=feature, pickle_dir='pickle_file/features.p')
    save_pickle(data=pca_add_feature, pickle_dir='pickle_file/pca_add_feature.p')

    print(f'train set {Counter(train_count)}')
    print(f'test set {Counter(test_count)}')
    # train set Counter({9: 1925, 3: 984, 8: 976, 1: 975, 7: 972, 4: 969, 2: 964, 6: 964, 5: 962, 0: 959})
    # test set Counter({9: 504, 0: 256, 5: 246, 2: 245, 6: 245, 7: 239, 4: 239, 8: 238, 1: 227, 3: 224})
    return


if __name__ == '__main__':
    make_train_set(dataset_dir='../CWRU')
