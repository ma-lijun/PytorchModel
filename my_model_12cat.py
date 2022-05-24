from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
# import cv2
import skimage.io as skiio
import imageio
import os
from torchvision import models, transforms
import time
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


num_categories = 12
# train images path
images_folder_train = ''
# train txt path
txt_train = r'G:\Users\AiStudio_cat12\train_list.txt'
# file bath path
bath_path = r'G://Users//AiStudio_cat12'

# valid images path
images_folder_val = ''

# device = torch.device('cpu')
device = torch.device('cuda:0')

image_extentions = ['.png', '.PNG', '.jpg', '.JPG']

def default_loader(path):
    return Image.open(path).convert('RGB')

default_transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ]
)

normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

normalize_dataset = transforms.Normalize(
    mean=[0.463, 0.400, 0.486],
    std=[0.191, 0.212, 0.170]
)

INPUT_SIZE = (448, 448)

class MyImageDataset(Dataset):

    def __init__(self, images_folder='', txt_path=txt_train, path_pre=os.getcwd(), transform=default_transform, loader=default_loader):
        images = []
        labels = []
        if images_folder:
            for dirname in os.listdir(images_folder):
                for filename in os.listdir(images_folder+dirname):
                    if any(filename.endswith(extention) for extention in image_extentions):
                        images.append((dirname+'\\'+filename, int(dirname)))
            self.images = images
        elif txt_path:
            fh = open(txt_path, 'r')
            imgs = []
            for line in fh:
                line = line.strip('\n')
                line = line.strip()
                # image_path = bath_path+os.sep+line.split()[0]
                image_path = os.path.join(bath_path, line.split()[0])
                # print("image_path: ", image_path, line)
                image_label = int(line.split()[1])
                # imgs_Norm = list(zip(image_path, image_label))
                # line.replace(" ", ",")
                # location = line.find('/')
                imgs.append(tuple((image_path, image_label)))
                # imgs.append((line, int(line[location + 1:location + 4]) - 1))

            # self.imgs = imgs
            self.images = imgs

        self.images_folder = images_folder
        self.transforms = transform
        self.txt_path = txt_path
        self.path_pre = path_pre
        self.loader = loader

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename, label = self.images[index]
        # if self.images_folder:
        #     img = Image.open(os.path.join(self.images_folder, filename)).covert('RGB')
        # elif self.txt_path:
        #     # path, label = self.imgs[index]
        #     img = self.loader(self.path_pre + filename)
        #     # img = self.loader(self.path_pre + path)
        # img = self.loader(self.path_pre + filename)
        # img = Image.open(os.path.join(self.images_folder, filename)).covert('RGB')
        # image = Image.open(filename)
        # image.show()
        # img = Image.open(filename)
        # img_pil_1 = np.array(img_pil)
        #
        # _img = skiio.imread(filename)

        # _img = Image.open(filename)
        # _img2 = _img.covert('RGB')
        img = imageio.imread(filename)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        
        
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.RandomCrop(INPUT_SIZE)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        # img = self.transforms(_img2)
        return img, label


train_data = MyImageDataset(txt_path=txt_train, path_pre=bath_path, transform=default_transform)
# test_data = MyImageDataset(images_folder_train, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4)

dataset_size = train_data.__len__()

# 官方推荐模型加载方法（只恢复模型参数）
# model = ModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
#
# #example
# resnet=resnet50(pretrained=True)
# resnet.load_state_dict(torch.load('ckp/model.pth'))

def initialize_model(model_name, num_categories, finetuning=False, pretrained=True, model_path=None):

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if finetuning == True:
            pass
            # 如果训练好的模型已经存在只需加载，不需要下载
            # if model_path:
            #     pre_dic = torch.load(r"G:\Users\pytorch_hub_checkpoints/resnet18-f37072fd.pth")
            #     Low_rankmodel_dic = model.state_dict()
            #     pre_dic = {k: v for k, v in pre_dic.items() if k in Low_rankmodel_dic}
            #     Low_rankmodel_dic.update(pre_dic)
            #     model.load_state_dict(Low_rankmodel_dic)
            # else:
            #     pass
        else:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        if finetuning == True:
            pass
            # 如果训练好的模型已经存在只需加载，不需要下载
            # if model_path:
            #     pre_dic = torch.load(r"G:\Users\pytorch_hub_checkpoints/resnet34-b627a593.pth")
            #     Low_rankmodel_dic = model.state_dict()
            #     pre_dic = {k: v for k, v in pre_dic.items() if k in Low_rankmodel_dic}
            #     Low_rankmodel_dic.update(pre_dic)
            #     model.load_state_dict(Low_rankmodel_dic)
            # else:
            #     pass
        else:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
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
                # scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # inputs, labels = inputs.to(device), targets.to(device)
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
            #  根据警告修改顺序：In PyTorch 1.1.0 and later, you should call them in the opposite order:
            #       `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this
            #       will result in PyTorch skipping the first value of the learning rate schedule.
            if phase == 'train':
                scheduler.step()

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
        cur_time_elapsed = time.time() - since
        print('Epoch {:.0f}s：Training complate in {:.0f}m {:.0f}s'.format(epoch, cur_time_elapsed // 60, cur_time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complate in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    checkpoint = torch.load('./checkpoints/state_best.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


model = initialize_model('resnet18', num_categories, model_path=True)
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# 每隔7个epoch学习率下降一次
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
pre_epoch = 0

model = train_model(model, criterion, optimizer, exp_lr_scheduler, pre_epoch, 10)


