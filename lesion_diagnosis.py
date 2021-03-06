# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from glob import glob
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import confusion_matrix


# CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define architecture
        self.features = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=2, bias=False, padding=2),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 32, kernel_size=5, stride=2, bias=False, padding=2),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False, padding=2),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU())

        # Classifier
        self.gap = nn.AvgPool2d(9, 7)
        self.classifier = nn.Linear(32, 7)

    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(self.gap(x))
        x = x.view(-1, x.size(0))
        x = self.classifier(x)
        return x


# Custom data loader
class HAMDataset(Dataset):
    def __init__(self, data_path, type):
        data = pd.read_csv(os.path.join(data_path, 'HAM10000', 'data.csv')).sort_values(by='image_id').reset_index(drop=True)
        # only get data for type (training or validation)
        dev_im_ids = pd.read_csv(os.path.join(data_path, 'val_split_info.csv'))
        if type is 'training':
            dev_indices = data.image_id.isin(dev_im_ids.image)[lambda x: x]
            data = data.drop(dev_indices.index).reset_index(drop=True)
        elif type is 'validation':
            train_indices = data.image_id.isin(dev_im_ids.image)[lambda x: ~x]
            data = data.drop(train_indices.index).reset_index(drop=True)

        # get labels
        self.labels = pd.factorize(data.dx)[0]
        self.label_names = pd.factorize(data.dx)[1]
        # get images
        self.im_paths = []
        for i in range(data.shape[0]):
            self.im_paths.append(os.path.join(data_path, 'HAM10000', data.image_id[i] + '.jpg'))
        # preprocessing & data augmentation
        self.composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # get image
        # x = np.expand_dims(np.asarray(Image.open(self.im_paths[idx]).convert('L')), 2)
        # x = np.asarray(Image.open(self.im_paths[idx]))
        # open image and crop centered
        x = transforms.functional.crop(Image.open(self.im_paths[idx]), 0, 75, 450, 450)
        #x = np.expand_dims(np.asarray(x), 2)
        # get label
        y = self.labels[idx]
        # preprocessing & data augmentation
        x = self.composed(x)
        return x, y


# Training function
def train(model, device, train_loader, optimizer, loss_function, epoch):
    # Set to training mode
    model.train()
    # Loop over all examples
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push to GPU
        data, target = data.to(device), target.to(device)
        # Reset gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Calculate outputs
            output = model(data)
            # Calculate loss
            loss = loss_function(output, target)
            # Backpropagate loss
            loss.backward()
            # Apply gradients
            optimizer.step()
        if batch_idx % 50 == 0:
            print("Train Epoch:", epoch, "Loss:", loss.item())


# Testing function
def test(model, device, test_loader, loss_function):
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            curr_loss = loss_function(output, target)
            if i == 0:
                predictions = output
                targets = target.data.cpu().numpy()
                loss = np.array([curr_loss.data.cpu().numpy()])
            else:
                predictions = np.concatenate((predictions, output))
                targets = np.concatenate((targets, target.data.cpu().numpy()))
                loss = np.concatenate((loss, np.array([curr_loss.data.cpu().numpy()])))
    # One-hot to normal:
    predictions = np.argmax(predictions, 1)
    # Caluclate metrics
    accuracy = np.mean(np.equal(predictions, targets))
    conf_mat = confusion_matrix(targets, predictions)
    sensitivity = conf_mat.diagonal() / conf_mat.sum(axis=1)
    # Print metrics
    print("Test Accuracy", accuracy, "Test Sensitivity", np.mean(sensitivity), "Test loss", np.mean(loss))


if __name__ == '__main__':
    # get dataset for testing and validation
    train_set = HAMDataset('data', 'training')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True)
    dev_set = HAMDataset('data', 'validation')
    dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False, pin_memory=True)
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define loss
    loss_function = nn.CrossEntropyLoss()
    # model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train
    for epoch in range(1, 50 + 1):
        train(model, device, train_loader, optimizer, loss_function, epoch)
        test(model, device, dev_loader, loss_function)

