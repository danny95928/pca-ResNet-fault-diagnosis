import matplotlib.pyplot as plt
import pickle


def load_pickle(pickle_file):
    open_file = open(pickle_file, 'rb')
    pickle_data = pickle.load(open_file)
    return pickle_data


def plot_curve(train_acc, index):

    plt.plot(train_acc, linewidth=3, color='red')
    plt.hlines(train_acc[index-1], xmin=0, xmax=index, color='green', ls='--', linewidth=1.5)
    plt.vlines(index, ymin=0, ymax=train_acc[index-1], color='green', ls='--', linewidth=1.5)
    plt.text(index, round(train_acc[index], 2) + 0.03, "%0.3f" % (train_acc[index]),
             color='blue', fontsize=8, va='center', ha='center')
    plt.title('Proportion of components in different dimensions')
    plt.ylabel('component')
    plt.xlabel('dims')
    plt.savefig('Proportion of components in different dimensions.png')


if __name__ == '__main__':
    path = 'pca_k_choice.p'
    data = load_pickle(pickle_file=path)
    plot_curve(data, index=256)