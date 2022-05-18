import os
from torchvision import transforms
import torchvision.models as models
from alexnet_pre_new import BuildAlexNetSTN
import skimage.io as io
import torch
from my_googlenet_pre_new import googlenet, MyGoogLeNet
# from my_googlenet_pre_new import MyGoogLeNet
from my_mobilenetv3_pre_new import MyMobileNetV3, my_mobilemodel
from my_resnet18_pre_new import MyResNet, BasicBlock, Bottleneck
from my_resnet18_ECA import MyEcaResNet,MyEcaResNet2, ECABasicBlock
from my_bilinear_cnn import BCNN



# paper_code_models_compare_googlenet_pre_92.14_2021-10-29_1749__2021-10-29_1749.pt
# paper_code_models_compare_googlenet_pre_95.95_2021-10-28_0933_model.pkl
# paper_code_stn_image_alexnet_pre_stn_pos1_98.16_2021-11-11_1740.pt_2021-11-11_1739.pt
# paper_code_models_compare_mobilenetv3_pre_2021-11-08_1018_98.04__2021-11-08_1018.pt
# paper_code_models_compare_resnet34_pre_2021-11-03_1632_97.85__2021-11-03_1632.pt
# paper_code_models_compare_alexnet_new_bn_2021-11-13_1509_96.32__2021-11-13_1509.pt
# paper_code_models_compare_bcnn_bilinear_cnn_pre_r18_2021-11-17_0954_63.89__2021-11-17_0954.pt
# paper_code_models_compare_new_folder_resnet18_pre_eca_second_train_2021-11-25_2022_62.5__2021-11-25_2022.pt
model_list = [
        # {"model": BCNN, "model_name": "bcnn_new", "file_name": "paper_code_models_compare_bcnn_bilinear_cnn_pre_r18_2021-11-17_0954_63.89__2021-11-17_0954.pt"},
        # {"model": BuildAlexNetSTN, "model_name": "alexnet_new_bn", "file_name": "paper_code_models_compare_alexnet_new_bn_2021-11-13_1509_96.32__2021-11-13_1509.pt"},
        # {"model": BuildAlexNetSTN, "model_name": "alexnet_pre", "file_name": "paper_code_models_compare_alexnet_pre_2021-11-12_0856_96.93__2021-11-12_0856.pt"},
        # {"model": BuildAlexNetSTN, "model_name": "alexnet_pre_stn", "file_name": "paper_code_stn_image_alexnet_pre_stn_pos1_98.16_2021-11-11_1740.pt_2021-11-11_1739.pt"},
        #           {"model":googlenet, "model_name":"googlenet_pre", "file_name":"paper_code_models_compare_googlenet_pre_95.95_2021-10-28_0933_model.pkl"},
        #           {"model":models.GoogLeNet, "model_name":"googlenet_pre", "file_name":"paper_code_models_compare_googlenet_pre_92.14_2021-10-29_1749__2021-10-29_1749.pt"},
        #           {"model":MyGoogLeNet, "model_name":"googlenet_pre", "file_name":"paper_code_models_compare_googlenet_pre_92.14_2021-10-29_1749__2021-10-29_1749.pt"},
                  # {"model":MyGoogLeNet, "model_name":"googlenet_new", "file_name":"paper_code_models_compare_googlenet_new_2021-11-08_1959_95.46__2021-11-08_1959.pt"},
                  # paper_code_models_compare_new_folder_resnet18_new_eca_2021-11-26_1211_77.11__2021-11-26_1211.pt
                  {"model":MyEcaResNet, "model_name":"resnet18_new_eca", "file_name":"paper_code_models_compare_new_folder_resnet18_new_eca_2021-11-26_1211_77.11__2021-11-26_1211.pt"},
                  {"model":MyEcaResNet, "model_name":"resnet18_pre_eca", "file_name":"paper_code_models_compare_new_folder_resnet18_pre_eca_2021-11-26_1211_82.78__2021-11-26_1211.pt"},
                  {"model":MyResNet, "model_name":"resnet18_pre", "file_name":"paper_code_models_compare_new_folder_resnet18_pre_2021-11-26_1211_83.31__2021-11-26_1211.pt"},
                  {"model":MyResNet, "model_name":"resnet18_new", "file_name":"paper_code_models_compare_new_folder_resnet18_new_2021-11-26_1211_74.77__2021-11-26_1211.pt"},
                  # {"model":MyEcaResNet, "model_name":"resnet18_pre2", "file_name":"paper_code_models_compare_resnet18_pre_eca_2021-11-25_1456_98.53__2021-11-25_1456.pt"},
                  # {"model":MyEcaResNet2, "model_name":"resnet18_pre_eca_second_train", "file_name":"paper_code_models_compare_new_folder_resnet18_pre_eca_second_train_2021-11-25_2022_62.5__2021-11-25_2022.pt"},
                  # {"model":my_mobilemodel, "model_name":"mobilev3_pre", "file_name":"paper_code_models_compare_mobilenetv3_pre_2021-11-12_1526_98.53__2021-11-12_1526.pt"},
                  # {"model":MyResNet, "model_name":"resnet34_pre", "file_name":"paper_code_models_compare_resnet34_pre_2021-11-12_1526_98.22__2021-11-12_1526.pt"},
                  # {"model":models.resnet50, "model_name":"resnet50_pre", "file_name":"paper_code_models_compare_resnet50_pre_98.34_2021-10-26_1809.pt_2021-10-26_1809.pt"},
                  # {"model":models.densenet121, "model_name":"densenet121_pre", "file_name":"paper_code_models_compare_densenet121_pre_98.28_2021-10-26_1809.pt_2021-10-26_1809.pt"},
        # {"model":resnet152, "model_name":"resnet152_pre"},
        # {"model": vgg16, "model_name": "vgg16_pre"},
    ]

base_path = r"G:\Users\experiment\pytorchmodel\model"
base_path2 = r"G:\Users\experiment\pytorchmodel\mycode"

li = [0 for i in range(55)]

import io

import torchvision.transforms as transforms
from PIL import Image

normalize_dataset_tomatoes = transforms.Normalize(
    mean=[0.4568, 0.4769, 0.4027],
    std=[0.1527, 0.1354, 0.1755]
)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)),
#
#     normalize_dataset_tomatoes
# ])

def transform_image(image_bytes):
    # my_transforms = transforms.Compose([transforms.Resize(255),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(
    #                                         [0.485, 0.456, 0.406],
    #                                         [0.229, 0.224, 0.225])])
    # normalize_dataset_tomatoes
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize_dataset_tomatoes])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# img_path = r"G:\Users\images\PlantDoc-Dataset-master\PlantDoc-Dataset-master\train"  # 查找文件的路径
# img_path = r"G:\Users\images\PlantDoc-Dataset-master\tomato"  # 查找文件的路径
# img_path = r"G:\Users\images\PlantDoc-Dataset-master\tomato_test1125"  # 查找文件的路径
img_path = r"C:\Users\zzsoft\ml\data\tomato_paper\images_validation"  # 查找文件的路径
# img_path = r"G:\Users\images\PlantDoc-Dataset-master\PlantDoc-Dataset-master\train\Tomato leaf yellow virus"  # 查找文件的路径
# img_path = r"G:\Users\images\PlantDoc-Dataset-master\PlantDoc-Dataset-master\train\Tomato leaf yellow virus"  # 查找文件的路径
# C:\Users\zzsoft\ml\data\images\valid_path2\images\0
# img_path = r"C:\Users\zzsoft\ml\data\images\valid_path2\images"  # 查找文件的路径

# img_path = r"G:\Users\images\PlantDoc-Dataset-master\PlantDoc-Dataset-master\train\Tomato leaf yellow virus"  # 查找文件的路径
# img_path = r"G:\Users\images\PlantDoc-Dataset-master\PlantDoc-Dataset-master\train"  # 查找文件的路径
total_num = 0
res_files_dict_li = []
# Tomato leaf bacterial spot
# Tomato leaf bacterial spot
# Tomato leaf late blight
# Tomato leaf mosaic virus
# Tomato mold leaf
# Tomato Septoria leaf spot
# Tomato leaf bacterial spot
disease_dict = {0:"Tomato leaf",
                1:"Tomato two spotted spider mites leaf",
                2:"Tomato Early blight leaf",
                3:"Tomato leaf late blight",
                4:"Tomato mold leaf",      # 叶霉病
                5:"Tomato Septoria leaf spot",
                6:"Tomato two spotted spider mites leaf",
                7:"Tomato leaf yellow virus",
                8:"Tomato leaf mosaic virus"
                }
for fpath, dirname, fnames in os.walk(img_path):
    if fpath == img_path:
        continue

    f_d = {"class_name":"", "f_label":0, "files":None}
    f_d["class_name"]=fpath.split("\\")[-1]
    # Tomato Bacterial Spot Bacteria general(番茄疮痂病）
    if f_d["class_name"]=="Tomato leaf bacterial spot":
        continue
    if f_d["class_name"]=="Tomato leaf":
        continue
    if f_d["class_name"]=="Tomato leaf mosaic virus":
        continue
    if f_d["class_name"]=="Tomato two spotted spider mites leaf":
        continue
    if f_d["class_name"]=="Tomato two spotted spider mites leaf":
        continue
    # if fpath.split("\\")[-1] == 'Tomato two spotted spider mites leaf':
    #     f_d["f_label"] = 6
    for k,v in disease_dict.items():
        if f_d["class_name"].strip() == v.strip():
            f_d["f_label"] = k

    total_num += len(fnames)
    # print(fpath, dirname, len(fnames), fnames)
    fnames = [fpath +os.sep + i for i in fnames]
    f_d["files"] = fnames
    res_files_dict_li.append(f_d)
    #     img_path = img_path +os.sep+ fnames[0]
print(total_num)

# c_img_path = r"C:\Users\zzsoft\ml\data\images\valid_path\images"
# dli = [{"disease_class": 2, "image_id": "fc38942e-e609-41a2-970c-cc1a67b16fe2___RS_Erly.B 7793.JPG"}, {"disease_class": 2, "image_id": "353aba987e85cc1f82ffb62d8cd39fb0.jpg"}, {"disease_class": 2, "image_id": "f7d118dea4625b2d38e41358869f8f60.jpg"}, {"disease_class": 2, "image_id": "e20fb98a-26f9-43a1-8d55-e1daa9b6a455___RS_Erly.B 7351.JPG"}, {"disease_class": 2, "image_id": "1d1d79c6-824d-4c48-bd59-c83a5cf29229___RS_Erly.B 9547.JPG"}, {"disease_class": 2, "image_id": "2c3651cb85a658bd9c8c6f6c86ccb9b8.jpg"}, {"disease_class": 2, "image_id": "3e7b2867350f9154476868cfcca62a04.jpg"}, {"disease_class": 2, "image_id": "73882c606eb644e1a8d5cc30dd54b51f.jpg"}, {"disease_class": 2, "image_id": "32e59188-5c46-4369-8991-5acfa253263b___RS_Erly.B 7781.JPG"}, {"disease_class": 2, "image_id": "42ae10468fe138264ec2c2f252138ffc.jpg"}, {"disease_class": 2, "image_id": "1863e110-8422-400b-89a6-f0ef457659cc___RS_Erly.B 6462.JPG"}, {"disease_class": 2, "image_id": "b6d8ce15-f786-4998-a67c-1390da696a07___RS_Erly.B 9521.JPG"}, {"disease_class": 2, "image_id": "0e03c87e-b43f-4cfe-a837-71306c68f4c0___RS_Erly.B 7733.JPG"}, {"disease_class": 2, "image_id": "96a2c52be93349f313980f2f8a0cd63e.jpg"}, {"disease_class": 2, "image_id": "654d7c11-664e-47c0-a6aa-c5c45a07e386___RS_Erly.B 7799.JPG"}, {"disease_class": 2, "image_id": "b58d7a66-097f-4167-abae-92f865407d90___RS_Erly.B 7845.JPG"}, {"disease_class": 2, "image_id": "b478689a7ad6eb412b01c87d4275f726.jpg"}, {"disease_class": 2, "image_id": "011aa278-a39e-40b0-b6e4-2e2f835889f0___RS_Erly.B 8251.JPG"}, {"disease_class": 2, "image_id": "5fb4b13612031ff61b5cb95d704d188b.jpg"}]
# fnames = [c_img_path+os.sep+i["image_id"] for i in dli]

# # Make sure to pass `pretrained` as `True` to use the pretrained weights:
# model = models.densenet121(pretrained=True)
# # Since we are using our model only for inference, switch to `eval` mode:
# model.eval()

# # todo mymodel
# model = BuildAlexNetSTN(9)
# model.load_state_dict(torch.load(base_path + os.sep + model_list[0]["file_name"]))

# model = models.googlenet(pretrained=True, num_classes=9)
# model.load_state_dict(torch.load(base_path2 + os.sep + model_list[1]["file_name"]))

for d in model_list:
    # if i == 0 or i >=2:
    #     continue
    kwargs = {"num_classes":20}
    # pretrained=True, num_classes=9
    # model = models.googlenet(pretrained=True, num_classes=9)
    # model = MyGoogLeNet(pretrained=True, num_classes=9, init_weights=False)
    # MyResNet
    if d.get("model_name") == "resnet18_pre":
        model = d["model"](BasicBlock, [2, 2, 2, 2], pretrained=True, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "resnet18_new":
        model = d["model"](BasicBlock, [2, 2, 2, 2], pretrained=False, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "resnet18_new_eca":
        model = d["model"](ECABasicBlock, [2, 2, 2, 2], pretrained=False, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "resnet18_pre_eca":
        model = d["model"](ECABasicBlock, [2, 2, 2, 2], pretrained=True, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "resnet18_pre2":
        model = d["model"](ECABasicBlock, [2, 2, 2, 2], pretrained=True, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "resnet18_pre_eca_second_train":
        model = d["model"](ECABasicBlock, [2, 2, 2, 2], pretrained=True, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "resnet34_pre":
        model = d["model"](BasicBlock, [3, 4, 6, 3], pretrained=True, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "alexnet_pre":
        model = d["model"](pretrained=True, num_classes=kwargs.get("num_classes"))
    elif d.get("model_name") == "alexnet_new_bn":
        model = d["model"](pretrained=True, num_classes=kwargs.get("num_classes"), BN_flag=True)
    elif d.get("model_name") == "alexnet_pre_stn":
        model = d["model"](pretrained=True, num_classes=kwargs.get("num_classes"), stn_flag=True)
    elif d.get("model_name") == "mobilenetv3_pre":
        model = d["model"](arch="mobilenet_v3_small", pretrained=True, progress=True, num_classes=9)
    else:
        # model = models.googlenet(pretrained=True, num_classes=9)
        model = d["model"](pretrained=True, num_classes=kwargs.get("num_classes"))

    # model = model_list[i]["model"](pretrained=True, num_classes=9)
    # print(torch.load(base_path + os.sep + model_list[i]["file_name"]))
    params_order_dict = torch.load(base_path + os.sep + d["file_name"])
    # print("params_order_dict:", params_order_dict)
    model.load_state_dict(params_order_dict)
    # model = torch.load(model_list[i]["file_name"])
    model.eval()
    # print(model)

    def get_prediction(image_bytes, model=model):
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        return y_hat


    # todo 此处才是实际应用
    # import json
    # imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))
    #
    # def get_prediction(image_bytes):
    #     tensor = transform_image(image_bytes=image_bytes)
    #     outputs = model.forward(tensor)
    #     _, y_hat = outputs.max(1)
    #     predicted_idx = str(y_hat.item())
    #     return imagenet_class_index[predicted_idx]

    # res_li = []
    acc = 0.0
    right_len = 0
    right_len_total = 0
    from collections import Counter
    for f_d in res_files_dict_li:
        f_li = f_d.get("files")
        f_label = f_d.get("f_label", 0)
        res_li = []
        result = {}
        for f_n in f_li:
            #   img_path +os.sep+ fnames[0]

            with open(f_n, 'rb') as f:
                image_bytes = f.read()
                try:
                    predict_res = get_prediction(image_bytes=image_bytes)
                except Exception as e:
                    print("f_d:", f_d)
                    print("f_n:", f_n)
                    print(e)
                    continue
                # print(get_prediction(image_bytes=image_bytes))
                res_li.append(predict_res.item())
        result = Counter(res_li)
        # if res_li and result.get(f_label,0):
        right_len = float(result.get(f_label, 0))
        print(f'modelname:{d["model_name"]}, label:{f_d["f_label"]}, right_len:{right_len}, class_len:{len(f_li)}, cur_acc:{right_len/len(f_li)},disease:{f_d["class_name"]}')
        right_len_total += right_len
    acc = right_len_total / total_num
    print("model name:{}, avg acc: {}%".format(d["model_name"], round(acc*100,2)))
    print("-"*50)


# D:\anaconda3\python.exe G:/Users/experiment/pytorchmodel/mycode/paper_code_models_compare.py
# BuildAlexNetSTN(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (4): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (5): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (6): ReLU(inplace=True)
#     (7): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (8): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (10): ReLU(inplace=True)
#     (11): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (13): ReLU(inplace=True)
#     (14): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=9, bias=True)
#   )
# )
#
#  model:alexnet_new_bn, lr:0.0005,batch:8
#
# Process finished with exit code -1

# bilinear
