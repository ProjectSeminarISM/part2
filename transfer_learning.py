import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Config:
    def __init__(self, data_dir):
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.data_dir = data_dir
        self.ckpt_dir = os.path.join(data_dir, 'checkpoints')
        # Create a dictionary that contains the information of the images in both the training and validation set
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                               for x in ['train', 'val']}
        # Create a dictionary that contains the data loader
        self.dataloader = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=32, shuffle=True)
                           for x in ['train', 'val']}
        # Create a dictionary that contains the size of each dataset (training and validation)
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        # Get the class names
        self.class_names = self.image_datasets['train'].classes
        # Print out the results
        print("Class Names: {}".format(self.class_names))
        print("{} batches in the training set".format(len(self.dataloader['train'])))
        print("{} batches in the test set".format(len(self.dataloader['val'])))
        print("{} training images".format(self.dataset_sizes['train']))
        print("{} testing images".format(self.dataset_sizes['val']))


class Model:
    def __init__(self, config):
        self.dataloader = config.dataloader
        self.config = config
        # Load the ResNet
        self.model = torchvision.models.resnet18(pretrained=True)

        # Freeze all layers in the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the number of inputs of the last layer (or number of neurons in the layer preceeding the last layer)
        num_ftrs = self.model.fc.in_features
        # Reconstruct the last layer (output layer) to have seven classes
        self.model.fc = nn.Linear(num_ftrs, 7)
        self.model.class_to_idx = self.dataloader['train'].dataset.class_to_idx

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        """
        # Understand what's happening
        iteration = 0
        correct = 0
        for inputs, labels in self.dataloader['train']:
            if iteration == 1:
                break
            inputs = Variable(inputs)
            labels = Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            print("For one iteration, this is what happens:")
            print("Input Shape:", inputs.shape)
            print("Labels Shape:", labels.shape)
            print("Labels are: {}".format(labels))
            output = self.model(inputs)
            print("Output Tensor:", output)
            print("Outputs Shape", output.shape)
            _, predicted = torch.max(output, 1)
            print("Predicted:", predicted)
            print("Predicted Shape", predicted.shape)
            correct += (predicted == labels).sum()
            print("Correct Predictions:", correct)
            iteration += 1
        """

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train(self):
        num_epochs = 2
        best_acc = 0.0
        df = pd.DataFrame()
        for epoch in range(num_epochs):
            self.exp_lr_scheduler.step()
            # Reset the correct to 0 after passing through all the dataset
            correct = 0
            for images, labels in self.dataloader['train']:
                images = Variable(images)
                labels = Variable(labels)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                # save to DataFrame
                df.append(pd.Series(np.apply_along_axis(self.softmax, 1, outputs.cpu().detach().numpy())[0]),
                               ignore_index=True)

            train_acc = 100 * correct.item() / self.config.dataset_sizes['train']
            print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {}%'
                  .format(epoch + 1, num_epochs, loss.item(), train_acc))
            # save model for every epoch
            print('[*] Saving model to {}'.format(self.config.ckpt_dir))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'class_to_idx': self.model.class_to_idx,
                'loss': loss,
            }, os.path.join(self.config.ckpt_dir, 'ckpt.pth.tar'))
            # change idx to class names
            idx_to_class = {val: key for key, val in self.model.class_to_idx.items()}
            df_classes = df.rename(idx_to_class, axis=1)
            df_classes.to_csv(os.path.join(self.config.data_dir, 'output', 'outputs.csv'))
            # distinguish best model and save separately
            if train_acc > best_acc:
                print('[*] ==== Best Acc Achieved ====')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'class_to_idx': self.model.class_to_idx,
                    'loss': loss,
                }, os.path.join(self.config.ckpt_dir, 'best_model.pth.tar'))
                df_classes.to_csv(os.path.join(self.config.data_dir, 'output', 'best_outputs.csv'))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (images, labels) in self.dataloader['val']:
                images = Variable(images)
                labels = Variable(labels)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy: {:.3f} %'.format(100 * correct / total))

    def show_img(self):
        # Visualize some predictions
        fig = plt.figure()
        shown_batch = 0
        index = 0
        with torch.no_grad():
            for (images, labels) in self.dataloader['val']:
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
                    ax.set_title('Predicted Label: {}'.format(self.config.class_names[preds[i]]))
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
    c = Config('data/lesion_data_multiclass')
    m = Model(c)
    m.train()
    m.test()
