import imghdr
import os
from PIL import Image

bath_path = r'G:\Users\AiStudio_cat12'

txt_train = bath_path + r'\train_list.txt'

cat_12_train = "cat_12_train"
cat_12_test = "cat_12_test"


good_imgs = []
bad_imgs = []

good_imgs_test = []
bad_imgs_test = []
# with open('work/img_list.txt', 'r') as f:
with open(txt_train, 'r') as f:
    paths = f.readlines()
    for i in range(len(paths)):
        img_path = os.path.join(bath_path, paths[i].split('\t')[0])
        if os.path.exists(img_path) and imghdr.what(img_path):
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(img_path)
            good_imgs.append(paths[i])# good image
        else:
            bad_imgs.append(paths[i])# bad image

print(len(good_imgs))
print("bad_imgs: ", len(bad_imgs))
print(bad_imgs)

# 获取文件夹 文件列表
test_li = os.listdir(bath_path+os.sep+cat_12_test)
# with open(txt_train, 'r') as f:
#     paths = f.readlines()
# for path in range(len(test_li)):
for path in test_li:
    # img_path = os.path.join(bath_path, paths[i].split('\t')[0])
    img_path = bath_path+os.sep+cat_12_test + os.sep + path
    if os.path.exists(img_path) and imghdr.what(img_path):
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(img_path)
        good_imgs_test.append(paths[i])# good image
    else:
        bad_imgs_test.append(paths[i])# bad image

print(len(good_imgs_test))
print("bad_imgs: ", len(bad_imgs_test))
print(bad_imgs_test)


def check_one_img(img_path):
    if os.path.exists(img_path) and imghdr.what(img_path):
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(img_path)
        good_imgs_test.append(paths[i])  # good image
        return True
    else:
        bad_imgs_test.append(paths[i])  # bad image
        return False

