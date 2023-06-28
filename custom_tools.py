import torch
import random
import numpy as np
from itertools import cycle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, savename, title, classes):  # 画混淆矩阵
    plt.figure(figsize=(15, 10), dpi=100)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(y_true, y_pred)
    # 在混淆矩阵中每格的概率值
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=15, va='center', ha='center')
        # plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title, fontsize=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=0, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.close()


def plot_curve(epoch_list, train_loss, train_acc, test_acc, savename, title):

    epoch = epoch_list
    plt.subplot(2, 1, 1)
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, test_acc, label="test_acc")

    plt.title('{}'.format(title))
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(epoch, train_loss, label="train_loss")
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))


def auc_show(target, pre, title_savename, n_class):
    classes = []
    for i in range(n_class):
        classes.append(i)
    target = label_binarize(target, classes=classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    plt.figure()

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}'.format(title_savename))
    plt.legend(loc="lower right")
    plt.savefig(r"result/{}.png".format(title_savename))
    plt.close()


class EarlyStopping():  # 提前停止
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=12, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_acc_min = np.Inf
        self.delta = delta

    def __call__(self, test_acc):

        score = test_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_acc)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter ----- > : {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_acc)
            self.counter = 0

    def save_checkpoint(self, test_acc):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'test acc ----> ({self.test_acc_min:.6f} --> {test_acc:.6f}).  Saving model ...')
        self.test_acc_min = test_acc
