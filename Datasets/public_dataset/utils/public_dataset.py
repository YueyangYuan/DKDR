from abc import abstractmethod
from argparse import Namespace
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import ImageFilter
import numpy as np
import random


class PublicDataset:
    NAME = None

    def __init__(self, args: Namespace, cfg, **kwargs) -> None:

        self.args = args
        self.cfg = cfg

    @abstractmethod
    def get_data_loaders(self) -> DataLoader:

        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:

        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:

        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:

        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    def random_loaders(self, train_dataset, public_len, public_batch_size):
        n_train = len(train_dataset)
        idxs = np.random.permutation(n_train)
        if public_len != None:
            idxs = idxs[0:public_len]
        train_sampler = SubsetRandomSampler(idxs)
        train_loader = DataLoader(train_dataset, batch_size=public_batch_size, sampler=train_sampler, num_workers=4)
        return train_loader


class ThreeCropsTransform:


    def __init__(self, transform):
        self.transform1 = transform[0]
        self.transform2 = transform[1]
        self.transform3 = transform[2]

    def __call__(self, x):
        q = self.transform1(x)
        k = self.transform2(x)
        v = self.transform2(x)
        return [q, k, v]


class FourCropsTransform:


    def __init__(self, transform):
        self.transform1 = transform[0]
        self.transform2 = transform[1]
        self.transform3 = transform[2]
        self.transform4 = transform[3]

    def __call__(self, x):
        q = self.transform1(x)
        k = self.transform2(x)
        u = self.transform3(x)
        v = self.transform4(x)
        return [q, k, u, v]


class GaussianBlur(object):


    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
