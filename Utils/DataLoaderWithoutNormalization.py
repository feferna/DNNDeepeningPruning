import os

import matplotlib.pyplot as plt
import pandas
from PIL import Image

import torch
from torchvision import transforms, datasets
import numpy as np

from sklearn.utils import shuffle

from torch.utils.data.sampler import SubsetRandomSampler


class ISIC2016(torch.utils.data.Dataset):
    def __init__(self, df_data, data_dir, transform=None):
        super().__init__()
        self.df = df_data
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        img_name = self.df['image'][id]
        img_label = self.df['class'][id].astype(float)

        img_path = os.path.join(self.data_dir, img_name + '.jpg')
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, img_label


def data_loader(dataset_root_path, dataset_name, batch_size):
    if dataset_name == "ISIC2016":
        train_loader, valid_loader, test_loader = load_ISIC2016(dataset_root_path, batch_size)
    elif dataset_name == "ChestXRay":
        train_loader, valid_loader, test_loader = load_ChestXRay(dataset_root_path, batch_size)
    elif dataset_name == "CIFAR10":
        train_loader, valid_loader, test_loader = load_CIFAR10(dataset_root_path, batch_size)
    elif dataset_name == "CIFAR100":
        train_loader, valid_loader, test_loader = load_CIFAR100(dataset_root_path, batch_size)

    return train_loader, valid_loader, test_loader


def load_ISIC2016(dataset_root_path, batch_size):
    data_path = dataset_root_path + "/ISIC2016/"

    # Create test dataframe
    test_df = pandas.read_csv(data_path + "Test_GroundTruth.csv")
                                          
    transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

    train_path = data_path + "train_images/"  # ISIC 2016
    test_path = data_path + "test_images/"
    train_set = ISIC2016(df_data=train_df, data_dir=train_path, transform=transform)
    test_set = ISIC2016(df_data=test_df, data_dir=test_path, transform=transform)

    dataset_len = len(train_set)
    indices = list(range(dataset_len))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True)

    return train_loader, None, test_loader


def load_ChestXRay(dataset_root_path, batch_size):
    data_path = dataset_root_path + "/chest_xray/"

    transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

    train_path = data_path + "train/"
    valid_path = data_path + "val/"
    test_path = data_path + "test/"

    train_set = datasets.ImageFolder(root=train_path, transform=transform)
    test_set = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    return train_loader, None, test_loader


def load_CIFAR10(dataset_root_path, batch_size):
    data_path = dataset_root_path + "/CIFAR10/"
    validation_split = 0.2

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_set = datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=True)
    test_set = datasets.CIFAR10(root=data_path, train=False, transform=transform_test, download=False)

    dataset_len = len(train_set)
    indices = list(range(dataset_len))

    # Randomly splitting indices:
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8)

    validation_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=8)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=8)

    return train_loader, validation_loader, test_loader


def load_CIFAR100(dataset_root_path, batch_size):
    data_path = dataset_root_path + "/CIFAR100/"
    validation_split = 0.2

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_set = datasets.CIFAR100(root=data_path, train=True, transform=transform_train, download=True)
    test_set = datasets.CIFAR100(root=data_path, train=False, transform=transform_test, download=False)

    dataset_len = len(train_set)
    indices = list(range(dataset_len))

    # Randomly splitting indices:
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8)

    validation_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=8)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=8)

    return train_loader, validation_loader, test_loader
