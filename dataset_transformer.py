import multiprocessing
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.manual_seed(73647)

# Creating a class to transform training and testing data
class CIFAR10_Transformer:
    def __init__(self, batch_size):
        # Initializing dataset and setting mean and std for the data
        self.batch_size = batch_size
        self.dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        self.mean,self.std = self.get_mean_and_std_for_dataset()
        self.cpu_count = min(multiprocessing.cpu_count(), 8)

    def get_mean_and_std_for_dataset(self):
        # Normalize data with range 0-255 over entire dataset
        data = self.dataset.data/255.0
        mean = data.mean(axis=(0, 1, 2))
        std = data.std(axis=(0, 1, 2))
        return mean,std

    def getTrainingData(self):
        # Setting transforms for training data
        # - horizontal flips on tha training data
        # - crop random parts in the image as input data
        # - convert to tensor
        # - normalize data over mean and std values
        train_transformers = [torchvision.transforms.RandomHorizontalFlip(),
                              torchvision.transforms.RandomCrop(size=32, padding=4),
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(mean=self.mean, std=self.std)]
        trainingDataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=torchvision.transforms.Compose(train_transformers))
        return DataLoader(trainingDataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)

    def getTestingData(self):
        # Setting transforms for testing data
        # - convert to tensor
        # - normalize data over mean and std values
        test_transformers = [torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize(mean=self.mean, std=self.std)]
        testingDataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.Compose(test_transformers))
        return DataLoader(testingDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.cpu_count)

