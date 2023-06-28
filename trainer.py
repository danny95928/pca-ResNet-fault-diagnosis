import torch.optim
from custom_tools import *
import time
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from model.BiGruModel import BiGruModel
from model.ResNetModel import ResNet
from model.FusionModel import ModalityFusion
from model.Fusion import Fusion
from batch_loader import ClassDataset
import torch.nn as nn
00000
import matplotlib.pyplot as plt  # todo 这两句是Ubuntu画图不报错
plt.switch_backend('agg')

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self,
                 learn_rate,
                 batch_size,
                 model_chose,
                 sigma):

        super(Trainer, self).__init__()

        chose = ['ResNet', 'FusionPca', 'BiGru', 'FusionFeature', 'FusionPcaAddFeature', 'Fusion']
        if model_chose not in chose:
            raise NameError("Model_combination should be one of {}, But you have chosen '{}', please correct it".
                            format(chose, model_chose))

        self.class_number = 10
        self.sigma = sigma
        self.batch_size = batch_size
        self.clip = -1
        self.lr = learn_rate
        self.model_chose = model_chose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.NLLLoss().to(self.device)

        if self.model_chose == 'ResNet':
            self.model = ResNet(num_classes=self.class_number)
        if self.model_chose == 'BiGru':
            self.model = BiGruModel(num_classes=self.class_number)
        if self.model_chose == 'FusionPca':
            self.model = ModalityFusion(num_classes=self.class_number, pca=True)
        if self.model_chose == 'FusionFeature':
            self.model = ModalityFusion(num_classes=self.class_number, feature=True)
        if self.model_chose == 'FusionPcaAddFeature':
            self.model = ModalityFusion(num_classes=self.class_number, pca_add_feature=True)
        if self.model_chose == 'Fusion':
            self.model = Fusion(num_classes=self.class_number)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model = self.model.to(self.device)

    def train(self, epoch, train_loader, train_len):

        self.model.train()
        correct = 0
        train_loss, train_acc, = 0, 0
        for batch_idx, batch in enumerate(train_loader):

            batch = [i.to(self.device) for i in batch]
            rnn_array, img_array, pca_array, feature_array, pca_add_feature_array, target = \
                batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

            if self.model_chose == 'ResNet':
                output = self.model(img_array)
            if self.model_chose == 'BiGru':
                output = self.model(rnn_array)
            if self.model_chose == 'FusionPca':
                output = self.model(rnn_array, img_array, pca_array)
            if self.model_chose == 'FusionFeature':
                output = self.model(rnn_array, img_array, feature_array)
            if self.model_chose == 'FusionPcaAddFeature':
                output = self.model(rnn_array, img_array, pca_add_feature_array)
            if self.model_chose == 'Fusion':
                output = self.model(rnn_array, img_array)

            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            if self.clip > 0:  # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            loss.backward()
            self.optimizer.step()
            train_loss += loss * target.size(0)
            argmax = torch.argmax(output, 1)
            train_acc += (argmax == target).sum()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.batch_size,
                    train_len, 100. * batch_idx * self.batch_size / train_len,
                    loss.item()))
        train_loss = torch.true_divide(train_loss, train_len)
        train_acc = torch.true_divide(train_acc, train_len)
        print('Train set: Average loss: {:.6f}, Accuracy: {}/{} ({:.5f}%)'.format(
            train_loss, correct, train_len, 100. * correct / train_len))
        return train_loss, train_acc

    def evaluate(self, epoch, test_loader, test_len, model_chose):
        global best_acc
        correct_test, test_acc = 0, 0
        # self.model.eval()
        with torch.no_grad():
            tar, argm = [], []
            for test_idx, test_batch in enumerate(test_loader):

                batch = [i.to(self.device) for i in test_batch]

                rnn_array, img_array, pca_array, feature_array, pca_add_feature_array, target = \
                    batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

                if self.model_chose == 'ResNet':
                    output = self.model(img_array)
                if self.model_chose == 'BiGru':
                    output = self.model(rnn_array)
                if self.model_chose == 'FusionPca':
                    output = self.model(rnn_array, img_array, pca_array)
                if self.model_chose == 'FusionFeature':
                    output = self.model(rnn_array, img_array, feature_array)
                if self.model_chose == 'FusionPcaAddFeature':
                    output = self.model(rnn_array, img_array, pca_add_feature_array)
                if self.model_chose == 'Fusion':
                    output = self.model(rnn_array, img_array)

                argmax = torch.argmax(output, 1)
                test_acc += (argmax == target).sum()
                pred_test = output.data.max(1, keepdim=True)[1]
                correct_test += pred_test.eq(target.data.view_as(pred_test)).cpu().sum()
                torch.cuda.empty_cache()

                tar.extend(target.cpu().numpy())
                argm.extend(argmax.cpu().numpy())

            test_acc = torch.true_divide(test_acc, test_len)

            print('\ntest set: Accuracy: {}/{} ({:.5f}%), Best_Accuracy({:.5f})'.format(
                correct_test, test_len, 100. * correct_test / test_len, best_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                print('The effect becomes better and the parameters are saved .......')
                weight = r'result/{}.pt'.format(model_chose)
                torch.save(self.model.state_dict(), weight)

                p = precision_score(tar, argm, average='macro')
                recall = recall_score(tar, argm, average='macro')
                f1 = f1_score(tar, argm, average='macro')

                plot_confusion_matrix(y_true=tar, y_pred=argm,
                                      savename=r"result/Confusion-Matrix-{}.png".format(model_chose),
                                      title=r"Confusion-Matrix-{}".format(model_chose),
                                      classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

                result_text = r'result/{}-text.txt'.format(model_chose)
                file_handle = open(result_text, mode='a+')
                file_handle.write('epoch:{},test_acc:{}, p:{}, recall:{},f1_score:{}\n'.format(
                    epoch, best_acc, p, recall, f1
                ))
                file_handle.close()
            return test_acc


def mian(learn_rate, batch_size, train_file, test_file, pca_file,
         feature_file, pca_add_feature_file, img_file, model_chose, patience, sigma):
    print(f'choose model name {model_chose}')
    print(f'use gpu {torch.cuda.get_device_name()}')

    train_set = ClassDataset(file=train_file,
                             pca_file=pca_file,
                             img_file=img_file,
                             feature=feature_file,
                             pca_add_feature=pca_add_feature_file,
                             sigma=sigma)

    test_set = ClassDataset(file=test_file,
                            pca_file=pca_file,
                            img_file=img_file,
                            feature=feature_file,
                            pca_add_feature=pca_add_feature_file,
                            sigma=sigma)

    train_len, test_len = len(train_set), len(test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    loss_train, acc_train, acc_test = [], [], []
    epoch_list = []
    start = time.time()

    T = Trainer(learn_rate, batch_size, model_chose, sigma)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(10000):
        epoch_list.append(epoch)
        train_loss, train_acc = T.train(epoch=epoch,
                                        train_loader=train_loader,
                                        train_len=train_len)

        test_acc = T.evaluate(epoch=epoch,
                              test_loader=test_loader,
                              test_len=test_len,
                              model_chose=model_chose)
        if torch.cuda.is_available():
            loss_train.append(train_loss.cuda().data.cpu().numpy())
            acc_train.append(train_acc.cuda().data.cpu().numpy())
            acc_test.append(test_acc.cuda().data.cpu().numpy())
        else:
            loss_train.append(train_loss.detach().numpy())
            acc_train.append(train_acc.detach().numpy())
            acc_test.append(test_acc.detach().numpy())

        early_stopping(test_acc)
        if early_stopping.early_stop:
            print(" === > Early stopping ! ! ! ! ! ")
            break
        print("....................... . Next . .......................")

    end = time.time()
    train_time = end - start
    print("训练时间长度为  ==== > {} s".format(train_time))

    plot_curve(epoch_list, loss_train, acc_train, acc_test,
               savename=r"result/train-loss-and-acc-{}".format(model_chose),
               title=r"train-loss-and-acc-{}".format(model_chose))


if __name__ == '__main__':
    best_acc = 0

    # chose = ['ResNet', 'FusionPca', 'BiGru', 'FusionFeature', 'FusionPcaAddFeature', 'Fusion']
    # sigma = [0, 5, 10, 15, ... ， 90, 95, 100]

    mian(learn_rate=0.001,
         batch_size=64,
         train_file=r'dataset/pickle_file/train.p',
         test_file=r'dataset/pickle_file/test.p',
         pca_file=r'dataset/pickle_file/pca_train.p',
         feature_file=r'dataset/pickle_file/features.p',
         pca_add_feature_file=r'dataset/pickle_file/pca_add_feature.p',
         img_file=r'dataset/IMAGE',
         model_chose='FusionPcaAddFeature',
         patience=12, sigma=10)
