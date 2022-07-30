from __future__ import print_function, division

import shutil

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
from filter_img import check_one_img

num_categories = 12
# train images path
images_folder_train = ''
# train txt path
# E:\data\cat12
# txt_train = r'E:\data\cat12\train_list.txt'
# txt_train = r'G:\Users\AiStudio_cat12\train_list.txt'
# file bath path
# G:\Users\AiStudio_cat12\cat_12_train
# bath_path = r'E:\data\cat12'
bath_path = r'G:\Users\AiStudio_cat12'

txt_train = bath_path + r'\train_list.txt'
txt_valid = bath_path + r'\valid_list.txt'

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
    # [transforms.Resize(ORIGIN_SIZE, Image.BILINEAR),
    [transforms.Resize(ORIGIN_SIZE),
     transforms.RandomCrop(INPUT_SIZE),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ]
)

valid_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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

train_parameters = {
    "class_dim": 12,  # 分类数
    "target_path": bath_path,
    'train_image_dir': bath_path + os.sep + 'trainImages',
    'eval_image_dir': bath_path + os.sep + 'evalImages',
    'test_image_dir': bath_path,
    'train_list_path': bath_path + os.sep + "train_self.txt",
    'eval_list_path': bath_path + os.sep + "valid_self.txt",
}


def create_train_eval():
    '''
    划分训练集和验证集
    '''
    train_dir = train_parameters['train_image_dir']
    eval_dir = train_parameters['eval_image_dir']
    train_list_path = train_parameters['train_list_path']
    target_path = train_parameters['target_path']
    print('creating training and eval images')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    with open(train_list_path, 'r') as f:
        data = f.readlines()
        for i in range(len(data)):
            img_path = data[i].split('\t')[0]
            class_label = data[i].split('\t')[1][:-1]
            if i % 8 == 0:  # 每8张图片取一个做验证数据
                eval_target_dir = os.path.join(eval_dir, str(class_label))
                eval_img_path = os.path.join(target_path, img_path)
                if not os.path.exists(eval_target_dir):
                    os.mkdir(eval_target_dir)
                shutil.copy(eval_img_path, eval_target_dir)
            else:
                train_target_dir = os.path.join(train_dir, str(class_label))
                train_img_path = os.path.join(target_path, img_path)
                if not os.path.exists(train_target_dir):
                    os.mkdir(train_target_dir)
                shutil.copy(train_img_path, train_target_dir)

    print('划分训练集和验证集完成！')


class MyImageDataset(Dataset):

    def __init__(self, images_folder='', txt_path=txt_train, path_pre=os.getcwd(), transform=default_transform,
                 loader=default_loader):
        images = []
        labels = []
        if images_folder:
            for dirname in os.listdir(images_folder):
                for filename in os.listdir(images_folder + dirname):
                    if any(filename.endswith(extention) for extention in image_extentions):
                        images.append((dirname + '\\' + filename, int(dirname)))
            self.images = images
        elif txt_path:
            fh = open(txt_path, 'r')
            imgs = []
            for line in fh:
                line = line.strip('\n')
                line = line.strip()
                image_path = os.path.join(bath_path, line.split()[0])
                check_flag = check_one_img(image_path)
                if check_flag:
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


# 官方推荐模型加载方法（只恢复模型参数）
# model = ModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
#
# #example
# resnet=resnet50(pretrained=True)
# resnet.load_state_dict(torch.load('ckp/model.pth'))


def initialize_model(model_name, num_categories, finetuning=False, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if finetuning == True:
            for p_name, param in model.named_parameters():
                if not p_name.startswith('layer4.1'):
                    param.requires_grad = False
            model.layer4[-1] = models.resnet.BasicBlock(512, 512)
        else:
            # for param in model.parameters():

            if finetuning:
                for p_name, param in model.named_parameters():
                    param.requires_grad = False
                model.layer4[-1] = models.resnet.BasicBlock(512, 512)
            else:
                for p_name, param in model.named_parameters():
                    param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        if finetuning == True:
            for p_name, param in model.named_parameters():
                if not p_name.startswith('layer4.1'):
                    param.requires_grad = False
            model.layer4[-1] = models.resnet.BasicBlock(512, 512)

        else:
            for p_name, param in model.named_parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        if finetuning == True:
            for p_name, param in model.named_parameters():
                if not p_name.startswith('layer4.2'):
                    param.requires_grad = False
            model.layer4[-1] = models.resnet.Bottleneck(2048, 512)

        else:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        # last_channel, num_classes
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_categories)
        model = model.to(device)
    else:
        model = None
    return model


def train_model(model, criterion, optimizer, scheduler, pre_epoch, num_epochs, model_name='resnet18'):
    since = time.time()
    best_acc = 0.0

    for epoch in range(pre_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # if phase == 'train':

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, labels = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                # torch.set_grad_enabled(False)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # torch.max(a, 1) 返回一行中最大值那个元素，且返回其索引（返回最大元素在这一行的列索引） 0为行索引
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #  根据警告修改顺序：In PyTorch 1.1.0 and later, you should call them in the opposite order:
                #       `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this
                #       will result in PyTorch skipping the first value of the learning rate schedule.
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            # epoch_acc = running_corrects.double() / dataset_size
            epoch_acc = float(running_corrects) / dataset_size

            print('{} loss: {:.4f} acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                checkpoint_path = 'state_best_' + model_name + '.tar'
                # checkpoint_path = 'state_best_score_'+'{:.4f}'.format(best_acc)+'.tar'
                torch.save({'epoch': epoch,
                            'model_name': model_name,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'acc': best_acc
                            }, checkpoint_path)
                print("saved best acc :{:.4f}".format(best_acc))
        cur_time_elapsed = time.time() - since
        print('Epoch {:.0f}s：Training complate in {:.0f}m {:.0f}s'.format(epoch, cur_time_elapsed // 60,
                                                                          cur_time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complate in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    checkpoint = torch.load('./state_best_' + model_name + '.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# predict function
def predict(input, model, device):
    # model.to(device)
    with torch.no_grad():
        input = input.to(device)
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
    # img = default_transform(img)
    # valid_transform
    img = valid_transform(img)
    # 扩展第一维度，bach * chanel * height * width
    img = img.unsqueeze(0)
    return img


if __name__ == '__main__':
    # model = initialize_model('resnet152', num_categories, finetuning=True)
    # model = initialize_model('resnet18', num_categories, finetuning=True)
    # todo 变量设置
    model_name = "mobilenet_v3_small"
    # para_dict = {"model_name": model_name, "opt_param": }
    model = initialize_model(model_name, num_categories)

    if model_name.startswith("res"):
        # 0.025 loss 不减
        optimizer = optim.SGD(model.fc.parameters(), lr=0.0025, momentum=0.9)
    elif model_name.startswith("mobile"):
        optimizer = optim.SGD(model.classifier[-1].parameters(), lr=0.0025, momentum=0.9)
    else:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.0025, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    # 每隔7个epoch学习率下降一次
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    pre_epoch = 0
    train_data = MyImageDataset(txt_path=txt_train, path_pre=bath_path, transform=default_transform)
    # test_data = MyImageDataset(images_folder_train, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4)

    dataset_size = train_data.__len__()

    # todo 模型恢复示例
    # model = ModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    #
    # #example
    # resnet=resnet50(pretrained=True)
    # todo 模型具有train()/eval() 两个方法，预测时要保证model在eval模式下
    # model.eval()
    # checkpoint = torch.load('./state_best_resnet152.tar')
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # todo train
    # model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, pre_epoch, 10, model_name='resnet152')
    # model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, 0, 30, model_name='resnet152')
    model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, 0, 30, model_name=model_name)

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
    test_data_dir = pathlib.Path(bath_path + r'\cat_12_test')
    # 不带目录，直接图片
    test_files = list(test_data_dir.glob('*.jpg'))
    for myfile in test_files:
        filename = os.path.basename(myfile)

        # cur_img = read_img(r'E:\data\cat12\cat_12_test', img_name=filename)
        # bath_path
        cur_img = read_img(bath_path + r'\cat_12_test', img_name=filename)

        # 使用训练结束后的模型，参数是被训练过的
        pre_res = predict(cur_img, model_trained, device)
        # pre_res = predict(cur_img, model, device)
        # print("img_name:{},pre: {}".format(filename, pre_res))
        # 写入文件
        # f.write(f"cat_12_test/{filename},{pre_res}\n")
        f.write(f"{filename},{pre_res}\n")
        print(f"{filename},{pre_res}\n")
    f.close()
