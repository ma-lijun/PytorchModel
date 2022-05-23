# encoding:utf-8
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import bilinear_model
from torch.autograd import Variable
import data
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainset = data.MyDataset('/Users/zhulingang/Desktop/新菊花数据集/new_train.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=4)

testset = data.MyDataset('/Users/zhulingang/Desktop/新菊花数据集/test.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=4)
cudnn.benchmark = True



model = bilinear_model.Net()
print (model)
model.cuda()

pretrained = True
if pretrained:

    pre_dic = torch.load("./vgg16-397923af.pth")
    Low_rankmodel_dic = model.state_dict()
    pre_dic = {k: v for k, v in pre_dic.items() if k in Low_rankmodel_dic}
    Low_rankmodel_dic.update(pre_dic)
    model.load_state_dict(Low_rankmodel_dic)




criterion = nn.CrossEntropyLoss()

model.features.requires_grad = False


optimizer = optim.SGD([
                       {'params': model.classifiers.parameters(), 'lr': 1.0}], lr=1, momentum=0.9, weight_decay=1e-5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 16., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))


def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


for epoch in range(1, 81):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    if epoch%5==0:
        test()
torch.save(model.state_dict(), 'bcnn_lastlayer.pth')
