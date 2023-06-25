import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#matplotlib inline
import torch.fx
import torch
import torch.nn as nn
import torch.optim as optim
from torchattacks import *
import torchvision.utils
from torchvision import models,datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchattacks
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import imshow, image_folder_custom_label
import cv2
def show_images_diff(original_img,original_label,adversarial_img,adversarial_label,eps):
    import matplotlib.pyplot as plt
    plt.figure()
    # plt.suptitle("FGSM Attack",x=0.5,y=0.8)
    #归一化
    if original_img.any() > 1.0:
        original_img=original_img/255.0
    if adversarial_img.any() > 1.0:
        adversarial_img=adversarial_img/255.0

    plt.subplot(131)
    plt.title(original_label)
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title(adversarial_label)
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    # plt.title('Eps={:.5f}'.format(eps))
    difference = adversarial_img - original_img
    plt.title('Eps={:.5f}'.format(float(abs(difference[0, 0, 0]))))
    #(-1,1)  -> (0,1)
    difference=difference / abs(difference).max()/2.0+0.5
    plt.imshow(difference,cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
#解析ImageNet的标签类，方便查看label对应的类别
class_idx = json.load(open("./data/imagenet_class_index.json"))#字典
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]#类名
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]#类名前面的一串数字
#图像预处理，进行resize是为了方便输入模型，totensor转换数据类型，此处不做normalization
#应该和模型有关，归一化不会改变图片，而normalization是会改变图片
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])
imagnet_data = datasets.ImageFolder(root='./data/imagenet', transform=transform)#此处可更改数据集，记住一定是只读取到文件夹上一层
data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
# iter是内置的迭代器
# images, labels = iter(data_loader).next()
#
# print("True Image & True Label")
# imshow(torchvision.utils.make_grid(images, normalize=True), [imagnet_data.classes[i] for i in labels])


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = nn.Sequential(
    norm_layer,
    models.resnet50(pretrained=True)
).to(device)

model = model.eval()

#将你想要的攻击函数注释掉
atks = [
    # FGSM(model, eps=20/225),
    BIM(model, eps=20/255, alpha=2/255, steps=8),#攻击成功率100%Total elapsed time (sec): 5061.56
    # MIFGSM(model, eps=8/255, steps=5, decay=1.0),
    # NIFGSM(model, eps=8/255, alpha=2/255, steps=5, decay=1.0),
    # PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)

]

def get_key(dct, key):
    return [v[1] for (k, v) in dct.items() if k == key]
print("Adversarial Image & Predicted Label")

for atk in atks:

    print("-" * 70)
    print(atk)

    correct = 0
    total = 0
    start = time.time()
    i=1
    accuracy=0
    for images, labels in data_loader:
        images = images.to(device)
        true_pre = np.argmax(model(images).data.cpu().numpy())
        labels = Variable(torch.Tensor([float(true_pre)]).long())
        labels = labels.to(device)

        adv_images = atk(images, labels)
        org = model(images)
        org_class = org.max(dim=1)[1].item()
        print("org label:", org_class)
        print("org class:", get_key(class_idx, str(org_class)))
        outputs = model(adv_images)
        y_adv = (model(adv_images)).cpu().detach().numpy()
        y_adv = np.argmax(y_adv, axis=1)
        print("y_adv={}".format(y_adv))
        print("adv class:", get_key(class_idx, str(y_adv.item())))
        if y_adv.item() != org_class:
            accuracy = accuracy + 1
            print("攻击成功！")
        else:
            print("攻击失败！")
        # 保存图片代码，需要的话注释掉
        adv = adv_images.cpu().detach().numpy()[0]
        adv = adv.transpose(1, 2, 0)
        save_adv = adv * 255.0
        save_adv= np.clip(save_adv, 0, 255).astype(np.uint8)
        save_adv = cv2.cvtColor(save_adv, cv2.COLOR_RGB2BGR)
        a = images.data.cpu().numpy()[0]
        a = a.transpose(1, 2, 0)
        show_images_diff(a,get_key(class_idx, str(org_class)), adv,get_key(class_idx, str(y_adv.item())),eps=0.2/255)#eps  需要自己手动设置，和攻击函数里的一样就行
        plt.show()
        # cv2.imwrite("D:\demo\\adversarial-attacks\\adv_pic\\FGSM{}.jpg".format(i), save_adv)#设置保存图片
        if i == 1000:  # 需要攻击多少图片
            print("一共攻击了{}张图片!".format(i))
            break
        i = i + 1
    print("========================")
    print("攻击成功率为：{}".format(accuracy / i))


