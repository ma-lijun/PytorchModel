from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
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
ORIGIN_SIZE = (600, 600)
INPUT_SIZE = (448, 448)

def default_loader(path):
    return Image.open(path).convert('RGB')

default_transform = transforms.Compose(
    [transforms.Resize(ORIGIN_SIZE, Image.BILINEAR),
     transforms.RandomCrop(INPUT_SIZE),
     transforms.RandomHorizontalFlip(),
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

normalize_train = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

normalize_dataset = transforms.Normalize(
    mean=[0.463, 0.400, 0.486],
    std=[0.191, 0.212, 0.170]
)


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
                image_path = os.path.join(bath_path, line.split()[0])
                image_label = int(line.split()[1])
                imgs.append(tuple((image_path, image_label)))

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
        img = imageio.imread(filename)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = self.transforms(img)
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
                # checkpoint_path = 'state_best_score_'+'{:.4f}'.format(best_acc)+'.tar'
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'acc': best_acc
                            }, checkpoint_path)
                print("saved best acc :{:.4f}".format(best_acc))
        cur_time_elapsed = time.time() - since
        print('Epoch {:.0f}s：Training complate in {:.0f}m {:.0f}s'.format(epoch, cur_time_elapsed // 60, cur_time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complate in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    checkpoint = torch.load('./state_best.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

#  todo 预测时候注销，初次训练使用?
model = initialize_model('resnet34', num_categories, model_path=True)
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# 每隔7个epoch学习率下降一次
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
pre_epoch = 0

# model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, pre_epoch, 10)

# predict function
def predict(input, model, device):
  # model.to(device)
  with torch.no_grad():
    input=input.to(device)
    out = model(input)
    _, pre = torch.max(out.data, 1)
    return pre.item()


# 将图片读取转换为 预测需要的tensor格式
def read_img(bath_path, img_name=""):
    """
    将图片读取转换为 预测需要的tensor格式
    :param bath_path:
    :param img_name:
    :return:
    """
    pre_img_path = os.path.join(bath_path, img_name)
    img = Image.open(pre_img_path)
    # img.show()
    img = img.convert('RGB')
    img = default_transform(img)
    # 扩展第一维度，bach * chanel * width * height
    img = img.unsqueeze(0)
    return img


if __name__ == '__main__':
    # todo 模型恢复示例
    # model = ModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    #
    # #example
    # resnet=resnet50(pretrained=True)
    # checkpoint = torch.load('./state_best.tar')
    # state_best_score_0.9083.tar
    # checkpoint = torch.load('./state_best_score_0.9083.tar')
    # model.load_state_dict(checkpoint['model_state_dict'])

    model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, pre_epoch, 15)


    # 单文件预测
    # pre_img_path = os.path.join(bath_path,"cat_12_train/i6duIYpPQTK0FgVa2eRBUCytcjEqr3v1.jpg")
    # img = Image.open(pre_img_path)
    # img = img.convert('RGB')
    # img = default_transform(img)
    # img = img.unsqueeze(0)
    # pre_res = predict(img, model, device)
    # print("pre: ", pre_res)

    # 批量预测
    import pathlib
    import os

    # 结果文件
    f = open("result.csv", "w")
    # f.write('image_id,label\n')
    # 遍历文件夹
    test_data_dir = pathlib.Path(r'G://Users//AiStudio_cat12//cat_12_test')
    # 不带目录，直接图片
    test_files = list(test_data_dir.glob('*.jpg'))
    for myfile in test_files:
        filename = os.path.basename(myfile)

        cur_img = read_img(r'G://Users//AiStudio_cat12//cat_12_test', img_name=filename)
        # 使用训练结束后的模型，参数是被训练过的
        pre_res = predict(cur_img, model_trained, device)
        print("img_name:{},pre: {}".format(filename, pre_res))
        # 写入文件
        f.write(f"cat_12_test/{filename},{pre_res}\n")
    f.close()
