from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import os
from torchvision import models, transforms
import time
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


num_categories = 2
# train images path
images_folder_train = ''
# valid images path
images_folder_val = ''

# device = torch.device('cpu')
device = torch.device('cuda:0')

image_extentions = ['.png', '.PNG', '.jpg', '.JPG']


class MyImageDataset(Dataset):

    def __init__(self, images_folder, transform=None):
        images = []
        labels = []
        for dirname in os.listdir(images_folder):
            for filename in os.listdir(images_folder+dirname):
                if any(filename.endswith(extention) for extention in image_extentions):
                    images.append((dirname+'\\'+filename, int(dirname)))

        self.images_folder = images_folder
        self.transforms = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename, label = self.images[index]
        img = Image.open(os.path.join(self.images_folder, filename)).covert('RGB')
        img = self.transforms(img)
        return img, label


transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

train_data = MyImageDataset(images_folder=images_folder_train, transform=transform)
test_data = MyImageDataset(images_folder_train, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4)

dataset_size = train_data.__len__()


def initialize_model(model_name, num_categories, finetuning=False, pretrained=True):

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if finetuning == True:
            pass
        else:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = models.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        if finetuning == True:
            pass
        else:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = models.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    else:
        model = None
    return model


def train_model(model, criterion, optimizer, scheduler, pre_epoch, num_epochs):
    since = time.time()
    best_acc = 0.0

    for epoch in range(pre_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, labels = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                # torch.set_grad_enabled(False)
                with torch.set_grad_enabled(phase== 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # torch.max(a, 1) 返回一行中最大值那个元素，且返回其索引（返回最大元素在这一行的列索引） 0为行索引
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss /dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('{} loss: {:.4f} acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc> best_acc:
                best_acc = epoch_acc
                checkpoint_path = 'state_best.tar'
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'acc': best_acc
                            }, checkpoint_path)
        print()

    time_elapsed = time.time() - since
    print('Training complate in {:.0f}m {:.0f}s'.format_map(time_elapsed//60, time_elapsed%60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    checkpoint = torch.load('./checkpoints/state_best.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


model = initialize_model('resnet34', num_categories)
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# 每隔7个epoch学习率下降一次
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
pre_epoch = 0

model = train_model(model, criterion, optimizer, exp_lr_scheduler, pre_epoch, 100)


