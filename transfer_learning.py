import torch
import torch.nn as nn
import torch.optim as optim
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
        # Data augmentation and normalization for training
        # Just normalization for validation
        """
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
        """
        self.data_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.data_dir = data_dir
        self.ckpt_dir = os.path.join(os.path.split(data_dir)[:-1][0], 'checkpoints')
        self.output_dir = os.path.join(os.path.split(data_dir)[:-1][0], 'output')
        # Create a dictionary that contains the information of the images in both the training and validation set
        """
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                               for x in ['train', 'val']}
        """
        self.image_dataset = datasets.ImageFolder(self.data_dir, self.data_transform)
        """
        # Create a dictionary that contains the data loader
        self.dataloader= {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4, shuffle=True)
                           for x in ['train', 'val']}
        """
        # Create a dictionary that contains the size of each dataset (training and validation)
        """
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        """
        self.dataset_size = len(self.image_dataset)
        # Get the class names
        self.class_names = self.image_dataset.classes
        # Print out the results
        print("Class Names: {}".format(self.class_names))
        # print("{} batches in the training set".format(len(self.dataloader['train'])))
        # print("{} batches in the test set".format(len(self.dataloader['val'])))
        print("{} training images".format(self.dataset_size))
        # print("{} testing images".format(self.dataset_sizes['val']))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        path, label = self.image_dataset.imgs[idx]
        sample = self.image_dataset.loader(path)
        sample = self.data_transform(sample)
        # return super(Config, self).__getitem__(idx), self.imgs[idx]
        return sample, label, path


class Model:
    def __init__(self, _train_loader, _val_loader, epochs):
        self.train_loader = _train_loader
        self.val_loader = _val_loader
        self.num_epochs = epochs

        # Load the ResNet
        self.model = torchvision.models.resnet18(pretrained=True)

        # Freeze all layers in the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the number of inputs of the last layer (or number of neurons in the layer preceeding the last layer)
        num_ftrs = self.model.fc.in_features
        # Reconstruct the last layer (output layer) to have seven classes
        self.model.fc = nn.Linear(num_ftrs, 7)
        # get class indices
        self.model.class_to_idx = self.val_loader.dataset.image_dataset.class_to_idx

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train(self):
        best_acc = 0.0
        # image_id = []
        # df = pd.DataFrame()
        for epoch in range(self.num_epochs):
            self.exp_lr_scheduler.step()
            # Reset the correct to 0 after passing through all the dataset
            correct = 0
            for images, labels, _ in self.train_loader:
                images = Variable(images)
                labels = Variable(labels)
                """
                for i in range(len(path)):
                    image_id.append(os.path.splitext(os.path.split(path[i])[-1])[0])
                """
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                """
                # save to DataFrame
                df = df.append(
                    pd.DataFrame(data=np.apply_along_axis(self.softmax, 1, outputs.cpu().detach().numpy())),
                    ignore_index=True)
                """

            train_acc = 100 * correct / len(self.train_loader.dataset)
            print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.4f}%'
                  .format(epoch + 1, self.num_epochs, loss.item(), train_acc))
            # save model for every epoch
            print('[*] Saving model to {}'.format(self.train_loader.dataset.ckpt_dir))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'class_to_idx': self.model.class_to_idx,
                'loss': loss,
            }, os.path.join(self.train_loader.dataset.ckpt_dir, 'ckpt.pth.tar'))
            """
            # change idx to class names, rename df, add image_ids, arrange correctly and sort by im_ids
            idx_to_class = {val: key for key, val in self.model.class_to_idx.items()}
            df_classes = df.rename(idx_to_class, axis=1)
            df_classes['image'] = image_id
            df_classes = df_classes[['image', 'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']]
            df_classes = df_classes.sort_values(by='image')
            df_classes.to_csv(os.path.join(self.config.data_dir, 'output', 'outputs.csv'), index=False)
            """
            # distinguish best model and save separately
            if train_acc > best_acc:
                print('[*] ==== Best Acc Achieved ====')
                best_acc = train_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'class_to_idx': self.model.class_to_idx,
                    'loss': loss,
                }, os.path.join(self.train_loader.dataset.ckpt_dir, 'best_model.pth.tar'))
                # df_classes.to_csv(os.path.join(self.config.data_dir, 'output', 'best_outputs.csv'), index=False)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            image_id = []
            df = pd.DataFrame()
            for idx, (images, labels, path) in enumerate(self.val_loader):
                images = Variable(images)
                labels = Variable(labels)
                for i in range(len(path)):
                    image_id.append(os.path.splitext(os.path.split(path[i])[-1])[0])
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.model(images)
                curr_loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if idx == 0:
                    predictions = predicted.data.cpu().numpy()
                    targets = labels.data.cpu().numpy()
                    loss = np.array([curr_loss.data.cpu().numpy()])
                else:
                    predictions = np.concatenate((predictions, predicted.data.cpu().numpy()))
                    targets = np.concatenate((targets, labels.data.cpu().numpy()))
                    loss = np.concatenate((loss, np.array([curr_loss.data.cpu().numpy()])))

                # save to DataFrame
                df = df.append(
                    pd.DataFrame(data=np.apply_along_axis(self.softmax, 1, outputs.cpu().detach().numpy())),
                    ignore_index=True)

            # change idx to class names, rename df, add image_ids, arrange correctly and sort by im_ids
            idx_to_class = {val: key for key, val in self.model.class_to_idx.items()}
            df_classes = df.rename(idx_to_class, axis=1)
            df_classes['image'] = image_id
            df_classes = df_classes[['image', 'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']]
            df_classes = df_classes.sort_values(by='image')
            df_classes.to_csv(os.path.join(self.train_loader.dataset.output_dir, 'prob_table.csv'), index=False)

            # Calculate metrics
            accuracy = np.mean(np.equal(predictions, targets))
            conf_mat = confusion_matrix(targets, predictions)
            sensitivity = conf_mat.diagonal() / conf_mat.sum(axis=1)

            # Print metrics
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
    train_dataset = Config('data/lesion_data_multiclass/train')
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    val_dataset = Config('data/lesion_data_multiclass/val')
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=True)
    m = Model(train_loader, val_loader, epochs=30)
    m.train()
    m.test()
