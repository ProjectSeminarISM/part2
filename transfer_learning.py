import torch
import torch.nn as nn
import torch.optim as optim
from progressbar import *
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageFile
from sklearn.metrics import confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Config(datasets.ImageFolder):
    def __init__(self, data_dir):
        # data augmentation and normalization
        self.data_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # directories
        self.data_dir = data_dir
        self.ckpt_dir = os.path.join(os.path.split(data_dir)[:-1][0], 'checkpoints')
        self.output_dir = os.path.join(os.path.split(data_dir)[:-1][0], 'output')
        # get image dataset
        self.image_dataset = datasets.ImageFolder(self.data_dir, self.data_transform)
        # get length of dataset
        self.dataset_size = len(self.image_dataset)
        # get class names
        self.class_names = self.image_dataset.classes
        # print out the results
        print("Class Names: {}".format(self.class_names))
        print("{} training images".format(self.dataset_size))

    def __len__(self):
        return self.dataset_size

    # custom getter to return path
    def __getitem__(self, idx):
        path, label = self.image_dataset.imgs[idx]
        sample = self.image_dataset.loader(path)
        sample = self.data_transform(sample)
        return sample, label, path


class Model:
    def __init__(self, _train_loader, _val_loader, epochs):
        self.train_loader = _train_loader
        self.val_loader = _val_loader
        self.num_epochs = epochs

        # Load the ResNet
        self.model = torchvision.models.resnet152(pretrained=True)

        # Freeze all layers in the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the number of inputs of the last layer
        num_ftrs = self.model.fc.in_features
        # Reconstruct the last layer (output layer) to have seven classes
        self.model.fc = nn.Linear(num_ftrs, 7)
        # get class indices
        self.model.class_to_idx = self.val_loader.dataset.image_dataset.class_to_idx
        # use gpu if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # define loss
        self.criterion = nn.CrossEntropyLoss()
        # define optimizer
        # self.optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        # Decay learning rate by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        # init progressbar
        self.widgets = ['Progress: ', Percentage(), ' ', Bar(marker='=', left='[', right=']'), ' ', ETA(), ' ',
                        FileTransferSpeed()]

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def save(self, epoch, loss, name):
        print('[*] Saving model to {}'.format(self.train_loader.dataset.ckpt_dir))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'loss': loss,
        }, os.path.join(self.train_loader.dataset.ckpt_dir, name))

    def load(self, name):
        checkpoint = torch.load(os.path.join(self.train_loader.dataset.ckpt_dir, name))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def train(self):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            self.exp_lr_scheduler.step()
            # Reset the correct to 0 after passing through all the dataset
            correct = 0
            # show progress in loop
            progress = ProgressBar(widgets=self.widgets, maxval=len(self.train_loader))
            progress.start()
            # repeat for all batches
            for idx, (images, labels, _) in enumerate(self.train_loader):
                images = Variable(images)
                labels = Variable(labels)
                # push to gpu
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                # get outputs of model
                outputs = self.model(images)
                # calculate loss
                loss = self.criterion(outputs, labels)
                loss.backward()
                # optimization step
                self.optimizer.step()
                # get labels in batch input
                _, predicted = torch.max(outputs, 1)
                # sum all correctly predicted labels
                correct += (predicted == labels).sum().item()
                # update progress
                progress.update(idx)
            progress.finish()
            # calculate accuracy
            train_acc = 100 * correct / len(self.train_loader.dataset)
            print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.4f}%'
                  .format(epoch + 1, self.num_epochs, loss.item(), train_acc))
            # save model for every epoch
            self.save(epoch, loss, 'ckpt.pth.tar')
            # save best model separately
            if train_acc > best_acc:
                print('[*] ==== Best Acc Achieved ====')
                best_acc = train_acc
                self.save(epoch, loss, 'best_model.pth.tar')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            # for saving the correlating image ids to the outputs
            image_id = []
            # for saving the probabilities
            df = pd.DataFrame()
            # show progress in loop
            progress = ProgressBar(widgets=self.widgets, maxval=len(self.val_loader))
            progress.start()
            # repeat for all batches
            for idx, (images, labels, path) in enumerate(self.val_loader):
                images = Variable(images)
                labels = Variable(labels)
                # get image ids from path
                for i in range(len(path)):
                    image_id.append(os.path.splitext(os.path.split(path[i])[-1])[0])
                # push to gpu
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                # get outputs
                outputs = self.model(images)
                # calculate loss
                curr_loss = self.criterion(outputs, labels)
                # get predicted labels
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # sum all correctly predicted labels
                correct += (predicted == labels).sum().item()
                # for evaluation issues save predictions, labels and losses in 1D array
                if idx == 0:
                    predictions = predicted.data.cpu().numpy()
                    targets = labels.data.cpu().numpy()
                    loss = np.array([curr_loss.data.cpu().numpy()])
                else:
                    predictions = np.concatenate((predictions, predicted.data.cpu().numpy()))
                    targets = np.concatenate((targets, labels.data.cpu().numpy()))
                    loss = np.concatenate((loss, np.array([curr_loss.data.cpu().numpy()])))
                # update progress
                progress.update(idx)
                # save to DataFrame
                df = df.append(
                    pd.DataFrame(data=np.apply_along_axis(self.softmax, 1, outputs.cpu().detach().numpy())),
                    ignore_index=True)
            progress.finish()
            # change indices to class names
            idx_to_class = {val: key for key, val in self.model.class_to_idx.items()}
            df_classes = df.rename(idx_to_class, axis=1)
            # add image ids
            df_classes['image'] = image_id
            # change order
            df_classes = df_classes[['image', 'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']]
            # sort by image id
            df_classes = df_classes.sort_values(by='image')
            # and save to csv file
            df_classes.to_csv(os.path.join(self.train_loader.dataset.output_dir, 'prob_table.csv'), index=False)

            # calculate metrics
            accuracy = np.mean(np.equal(predictions, targets))
            conf_mat = confusion_matrix(targets, predictions)
            sensitivity = conf_mat.diagonal() / conf_mat.sum(axis=1)

            # print metrics
            print("Test Accuracy", accuracy, "Test Sensitivity", np.mean(sensitivity), "Test loss", np.mean(loss))

    def show_img(self):
        # Visualize some predictions
        fig = plt.figure()
        shown_batch = 0
        index = 0
        with torch.no_grad():
            for images, labels, path in self.val_loader:
                if shown_batch == 1:
                    break
                shown_batch += 1
                images = Variable(images)
                labels = Variable(labels)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.model(images)  # The output is of shape (4,2)
                _, preds = torch.max(outputs, 1)  # The pred is of shape (4) --> [ 0,  0,  0,  1]

                for i in range(4):
                    index += 1
                    ax = plt.subplot(2, 2, index)
                    ax.axis('off')
                    ax.set_title('Predicted Label: {}'.format(self.val_loader.dataset.class_names[preds[i]]))
                    # Get the tensor of the image, and put it to cpu
                    input_img = images.cpu().data[i]
                    # If we have a tensor of shape (2,3,4) --> it becomes (3,4,2)
                    inp = input_img.numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    inp = std * inp + mean
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)
                    plt.show()


if __name__ == '__main__':
    # training data
    train_dataset = Config('data/lesion_data_multiclass/train')
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    # validation data
    val_dataset = Config('data/lesion_data_multiclass/val')
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=True)

    m = Model(train_loader, val_loader, epochs=100)
    # load existing model
    # m.load('best_model_lr01_adam_resnet152.pth.tar')
    # train new model
    m.train()
    # test model
    m.test()
