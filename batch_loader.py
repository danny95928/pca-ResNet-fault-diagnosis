import os
from PIL import Image
from torchvision import transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import pickle
import numpy as np


class ClassDataset(Dataset):
    def __init__(self,
                 file: Optional[str],
                 pca_file: Optional[str],
                 feature: Optional[str],
                 pca_add_feature: Optional[str],
                 img_file: Optional[str],
                 sigma: Optional[int] = 10
                 ):
        super(ClassDataset, self).__init__()
        self.sigma = sigma

        self.batch_data = self.load_pickle(file)
        self.index = {i: key for i, key in enumerate(self.batch_data.keys())}

        self.pca_data = self.load_pickle(pca_file)
        self.feature = self.load_pickle(feature)
        self.pca_add_feature = self.load_pickle(pca_add_feature)
        self.img_file = img_file

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])

    def __getitem__(self, index):
        key = self.index[index]

        data = self.batch_data[key]
        train, label = data[:-1], data[-1]

        rnn_train = torch.tensor(train.reshape(16, 64), dtype=torch.float32)

        label = int(torch.tensor(label))

        # pcp特征
        pca_data = self.pca_data[key]
        pca_train = torch.tensor(pca_data, dtype=torch.float32)

        # 手工特征
        feature = self.feature[key]
        features = torch.tensor(feature, dtype=torch.float32)

        # pcp特征 + 手工特征
        pca_add_feature = self.pca_add_feature[key]
        pca_add_features = torch.tensor(pca_add_feature, dtype=torch.float32)

        img_file = os.path.join(self.img_file, f'{key}.png')
        img_array = Image.open(img_file)
        img_array = img_array.convert('L')
        img_array = self.transforms(img_array)

        noise2 = np.random.normal(0, self.sigma, img_array.shape).astype(np.float32)
        noise2 = noise2 / 255.
        img_array = img_array + noise2

        return rnn_train, img_array, pca_train, features, pca_add_features, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def load_pickle(pickle_file):
        open_file = open(pickle_file, 'rb')
        pickle_data = pickle.load(open_file)
        return pickle_data


if __name__ == '__main__':

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    dataSwt = ClassDataset(file=r'dataset/pickle_file/test.p',
                           pca_file=r'dataset/pickle_file/pca_train.p',
                           feature=r'dataset/pickle_file/features.p',
                           pca_add_feature=r'dataset/pickle_file/pca_add_feature.p',
                           img_file=r'dataset/IMAGE',
                           sigma=10)
    dataloader = DataLoader(dataSwt, batch_size=32, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
        print(batch[5])
        break
