import numpy as np
import scipy.io as scio
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


def plot_curve(train_acc):
    plt.plot(train_acc)
    plt.title('Proportion of components in different dimensions')
    plt.ylabel('component')
    plt.xlabel('dims')
    plt.savefig('Proportion of components in different dimensions.png')


def save_pickle(data, pickle_dir):
    file = open(pickle_dir, 'wb')
    pickle.dump(data, file)
    file.close()


def load_mat_return_array(path, time_step=100, seq_len=1024):
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
        data.append(X[id: id + seq_len].reshape(seq_len))
    return np.array(data)


def draw_k_components(dataset_dir, dim=500):
    namelist = os.listdir(dataset_dir)
    c = np.zeros(shape=(1, 1024))
    for name in namelist:
        mat_path = os.path.join(dataset_dir, name)
        data_only = load_mat_return_array(mat_path)
        c = np.vstack((c, data_only))
    c = c[1:]
    sum_list = []
    for i in range(1, dim + 1):
        print(i)
        pca = PCA(n_components=i)
        pca.fit(c)
        n = pca.explained_variance_ratio_
        count = []
        for x in n:
            count.append(x)
        sum_list.append(sum(count))
    save_pickle(sum_list, 'pca_k_choice.p')
    plot_curve(sum_list)
    return


if __name__ == '__main__':
    draw_k_components(dataset_dir='../CWRU', dim=500)
