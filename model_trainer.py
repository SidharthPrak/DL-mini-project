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

class DataTrainer():
    def __init__(self, model, device, epochs, trainLoader, testLoader):
        self.epochs = epochs
        self.global_best_accuracy = 0.0
        self.global_best_accuracy_epoch = -1
        self.learning_rate = 0.01
        self.weight_decay = 0.0001
        self.model = model
        self.training_loss = []
        self.testing_loss = []
        self.training_accuracy = []
        self.testing_accuracy = []

        self.trainLoader = trainLoader
        self.testLoader = testLoader

        self.lossFunction = torch.nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = torch.optim.Adadelta(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.epochs, eta_min=self.learning_rate/10.0)

        self.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_trainable_parameters(self):
        print("Total Trainable Parameters : %s"%(self.trainable_parameters))

    def check_if_model_under_budget(self, budget):
        if self.trainable_parameters > budget:
            raise Exception("Model is not under budget.")
        else:
            print("Model is under budget.")

    def training_phase(self):
        loader = self.trainLoader
        self.model.train()
        self.optimizer.zero_grad()
        total_loss_per_epoch = 0.0
        total_corrects_per_epoch = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = self.model(images)
            loss = self.lossFunction(output, labels)
            predicted_labels = torch.argmax(output, dim=1)
            total_loss_per_epoch += loss.item()
            total_corrects_per_epoch += torch.sum(predicted_labels == labels).float().item()
            loss.backward()
            self.optimizer.step()
        epoch_loss = total_loss_per_epoch/len(loader.dataset)
        epoch_accuracy = total_corrects_per_epoch/len(loader.dataset)
        self.scheduler.step()
        self.training_loss.append(epoch_loss)
        self.training_accuracy.append(epoch_accuracy)

    def testing_phase(self):
        loader = self.testLoader
        self.model.eval()
        total_loss_per_epoch = 0.0
        total_corrects_per_epoch = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = self.model(images)
            loss = self.lossFunction(output, labels)
            predicted_labels = torch.argmax(output, dim=1)
            total_loss_per_epoch += loss.item()
            total_corrects_per_epoch += torch.sum(predicted_labels == labels).float().item()

        epoch_loss = total_loss_per_epoch/len(loader.dataset)
        epoch_accuracy = total_corrects_per_epoch/len(loader.dataset)
        self.testing_loss.append(epoch_loss)
        self.testing_accuracy.append(epoch_accuracy)
        if epoch_accuracy > self.global_best_accuracy:
            self.global_best_accuracy = epoch_accuracy
            self.model.saveModel()

    def train_model(self):
        for i in tqdm(range(self.epochs)):
            self.training_phase()
            self.testing_phase()
            self.global_best_accuracy = max(self.testing_accuracy)
            self.global_best_accuracy_epoch = np.argmax(self.testing_accuracy)

            print("Training Loss: %s, Testing Loss: %s, Training Accuracy: %s, Testing Accuracy: %s" \
                  %(self.training_loss[-1], self.testing_loss[-1], self.training_accuracy[-1], self.testing_accuracy[-1]))

    def get_highest_accuracy(self):
        print("Max Testing Accuracy: %s"%(self.global_best_accuracy))

    def plot_accuracy_and_loss_graphs(self):
        fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(20, 10))
        train_length = len(self.training_loss)
        plot1.plot(range(train_length), self.training_loss, '-', linewidth='3', label='Training Error')
        plot1.plot(range(train_length), self.testing_loss, '-', linewidth='3', label='Testing Error')
        plot1.grid(True)
        plot1.legend()

        plot2.plot(range(train_length), self.training_accuracy, '-', linewidth='3', label='Training Accuracy')
        plot2.plot(range(train_length), self.testing_accuracy, '-', linewidth='3', label='Testing Acuracy')
        plot2.annotate('max accuracy = %s'%(self.global_best_accuracy), xy=(self.global_best_accuracy_epoch, self.global_best_accuracy), xytext=(self.global_best_accuracy_epoch, self.global_best_accuracy+0.1), arrowprops=dict())
        plot2.grid(True)
        plot2.legend()

        fig.savefig("./train_test_graphs.png")