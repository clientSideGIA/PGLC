import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
import torchvision.utils as vutils
from torchvision import models, datasets, transforms
from collections import defaultdict, OrderedDict
from copy import deepcopy
import re
import copy
import time
import math
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import sys
import os
from PIL import Image

import  global_var 
device = global_var.get_device()


'''
Utils
-Image Utils
-Metrics
-Logger
'''

#Image Utils
def pick_targeted_class_images(dataset, class_index, transform):
    '''
    To pick up targeted class images randomly from dataset.
    '''
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    path_list = (images[labels == class_index]).tolist()
    images = []
    for path in path_list:
        img = transform(Image.open(path))
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        images.append(img)

    return torch.stack(images)


def pick_targeted_class_images_loader(loader, class_index, image_num=1):
    '''
    To pick up targeted class images randomly from data loader.
    '''
    if loader.batch_size != 1:
        raise ValueError("loader batch size should be 1")
    images = []
    total_num = 0
    for X, y in loader:
        samples = X[y == class_index]
        for i in range(len(samples)):
            images.append(samples[i])
            total_num += 1
            if total_num == image_num:
                break
        if total_num == image_num:
            break

    return torch.stack(images)


def pick_targeted_class_images_path_array(dataset, class_index):
    '''
    To pick up targeted class images save paths randomly from dataset.
    '''
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    path_list = (images[labels == class_index])

    return path_list


def cal_images_statistic(image_tensor):
    '''
    calculate batch images mean and variance.
    '''
    nch = image_tensor.shape[1]
    mean = image_tensor.mean([0, 2, 3])
    var = image_tensor.permute(1, 0, 2, 3).contiguous().view(
        [nch, -1]).var(1, unbiased=False)

    return mean, var


def image2tensor_path(path,transform=None):
    '''
    image2tensor based on path
    '''
    
    if not transform:
        transform = transforms.ToTensor()
    
    return transform(Image.open(path))


def image2tensor_pathlist(path_list, transform):
    '''
    image2tensor based on path list
    '''
    if type(path_list) == np.ndarray:
        path_list = path_list.tolist()
    images = []
    for path in path_list:
        img = transform(Image.open(path))
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        images.append(img)

    return torch.stack(images)

def image2tensor_PIL(image,transform=None):
    '''
    image2tensor based on PIL
    '''
    
    if not transform:
        transform = transforms.ToTensor()
    
    return transform(image)

def tensor2image_PIL(single_image_tensor):
    '''
    image2tensor based on tensor
    '''
    transform = transforms.ToPILImage()
    
    return transform(image_tensor)


def random_select_image(dataset, size=32):
    '''
    Randomly select images from dataset
    '''
    
    if len(dataset.data.shape) == 3:
        num, _, _ = dataset.data.shape
    else:
        num, _, _, _ = dataset.data.shape
        
    index = random.randint(0, num)
    single_input = dataset.data[index][np.newaxis, ...]
    if len(single_input.shape) == 3:
        single_input = single_input[..., np.newaxis]
        single_input = single_input.expand(-1, -1, -1, 3)
    single_input = torch.Tensor(single_input)
    single_input = single_input.permute(0, 3, 1, 2)/255
    single_input = transforms.Resize(size)(single_input)
    label = dataset.targets[index]
    label = torch.tensor(label)
    return single_input, label


def random_select_images(dataset, size=32, batch_size=4):
    '''
    Randomly select images from dataset
    '''
    
    if len(dataset.data.shape) == 3:
        num, _, _ = dataset.data.shape
    else:
        num, _, _, _ = dataset.data.shape

    index_array = np.random.permutation(num)[:batch_size]

    inputs = []
    labels = []
    for index in index_array:
        single_input = dataset.data[index][np.newaxis, ...]
        if len(single_input.shape) == 3:
            single_input = single_input[..., np.newaxis]
            single_input = single_input.expand(-1, -1, -1, 3)
        single_input = torch.Tensor(single_input)
        single_input = single_input.permute(0, 3, 1, 2)/255
        single_input = transforms.Resize(size)(single_input)
        label = dataset.targets[index]
        label = torch.tensor(label)
        inputs.append(single_input)
        labels.append(label.item())

    inputs = torch.stack(inputs).squeeze()
    labels = torch.Tensor(labels).to(torch.int64)

    return inputs, labels


def show_image(img, title=None):
    '''
    Show image, tensor and dim = 3
    '''
    img = img[0]
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    if not title:
        plt.title(str(title))
    plt.axis('off')
    plt.show()


def show_images(img, labels=None):
    '''
    Show images, tensor and dim = 4
    '''
    num = img.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(num):
        plt.subplot(math.ceil(num/8), 8, i + 1)
        plt.imshow(transforms.ToPILImage()(img[i]))
        if labels != None:
            plt.title("%d" % labels[i].item())
        plt.axis('off')

    plt.show()

def show_batch_images(batch):
    '''
    Show images, tensor and dim = 4
    '''
    plt.figure(figsize=(12, 8))
    for i in range(batch.shape[0]):
        plt.subplot(1, batch.shape[0], i + 1)
        plt.imshow(transforms.ToPILImage()(batch[i]))
        plt.axis('off')

    plt.show()
    

def show_rebuild_images(history, num=30, begin=0):
    '''
    Show images, history is a list of tensors with dim = 4
    '''
    plt.figure(figsize=(12, 8))
    for i in range(num):
        plt.subplot(math.ceil(num/10), 10, i + 1)
        plt.imshow(transforms.ToPILImage()(history[i+begin][0]))
        plt.title("iter=%d" % ((i+begin) * 100))
        plt.axis('off')

    plt.show()


def show_rebuild_batch_images(history, index, num=70, begin=0):
    '''
    Show images, history is a list of tensors with dim = 4
    '''
    plt.figure(figsize=(12, 8))
    for i in range(num):
        plt.subplot(int(num/10), 10, i + 1)
        plt.imshow(transforms.ToPILImage()(history[i+begin][index]))
        plt.title("iter=%d" % ((i+begin) * 100))
        plt.axis('off')

    plt.show()


# Metrics
def cal_loss(input, labels, model, device, target_index=[0], log=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    input = input.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        target_index_set = set(target_index)
        index_set = set(list(range(len(input))))
        no_target_index_set = list(index_set - target_index_set)
        print(len(no_target_index_set))

        target_index = torch.tensor(target_index, dtype=torch.long).to(device)
        no_target_index = torch.tensor(
            no_target_index_set, dtype=torch.long).to(device)

        yp = model(input[target_index])
        tagret_loss = criterion(yp, labels[target_index])

        yp = model(input[no_target_index])
        other_loss = criterion(yp, labels[no_target_index])

        yp = model(input)
        total_loss = criterion(yp, labels)

        if log:
            print("Target Loss:", tagret_loss.item())
            print("Other Loss:", other_loss.item())
            print("Total Loss:", total_loss.item())
        else:
            return tagret_loss, other_loss, total_loss


def cal_loss_batch(input, model, device, target_batch_index=[0], target_index=[0], log=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    # input = input.to(device)
    # labels = labels.to(device)
    with torch.no_grad():
        total_target_loss = 0.
        for target_batch_idx in target_batch_index:
            target_images, target_labels = input[target_batch_idx]
            target_images, target_labels = target_images.to(
                device), target_labels.to(device)
            for target_idx in target_index:
                yp = model(target_images[target_idx:target_idx+1])
                tagret_loss = criterion(
                    yp, target_labels[target_idx:target_idx+1])
                total_target_loss += tagret_loss.item()

        target_index_set = set(target_index)
        index_set = set(list(range(len(input[0][0]))))
        no_target_index = list(index_set - target_index_set)

        total_other_loss = 0.
        for target_batch_idx in target_batch_index:
            target_images, target_labels = input[target_batch_idx]
            target_images, target_labels = target_images.to(
                device), target_labels.to(device)
            for no_target_idx in no_target_index:
                yp = model(target_images[no_target_idx:no_target_idx+1])
                no_tagret_loss = criterion(
                    yp, target_labels[no_target_idx:no_target_idx+1])
                total_other_loss += no_tagret_loss.item()/(len(no_target_index))
                #total_other_loss += no_tagret_loss.item()/(len(target_labels)-len(target_index))

        for batch_idx in range(len(input)):
            if batch_idx not in target_batch_index:
                images, labels = input[batch_idx]
                images, labels = images.to(device), labels.to(device)
                yp = model(images)
                loss = criterion(yp, labels)
                total_other_loss += loss.item()

        total_loss = 0.
        for batch_idx in range(len(input)):
            images, labels = input[batch_idx]
            images, labels = images.to(device), labels.to(device)
            yp = model(images)
            loss = criterion(yp, labels)
            total_loss += loss.item()

        if log:
            print("Target Loss:", total_target_loss/len(target_index))
            print("Other Loss:", total_other_loss/len(input))
            print("Total Loss:", total_loss/len(input))

        else:
            return total_target_loss/len(target_index), total_other_loss/len(input), total_loss/len(input)


def cal_MSE(input, target):
    return F.mse_loss(input, target)


def cal_PSNR(input, target):
    if torch.max(input) > 128:
        max_pixel = 255.0
    else:
        max_pixel = 1.0

    return 20 * torch.log10(max_pixel/torch.sqrt(F.mse_loss(input, target)))

def cal_SSIM(input, target, window_size=11, window=None, size_average=True, full=False, val_range=None):
    import math

    def gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel=1):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            channel, 1, window_size, window_size).contiguous()
        return window

    img1 = input
    img2 = target

    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret



def cal_LPIPS(input, target):
    import lpips
    if 'lp_model' not in globals():
        global lp_model
        lp_model = lpips.LPIPS(net="vgg").to(device)
    
    if torch.max(input) > 128:
        input = input/255.0
        target = target/255.0

    return lp_model(input, target).mean()

class Metric:
    def __init__(self):
        self.device = device
        import lpips
        self.lp_model = lpips.LPIPS(net="vgg").to(device)
        
    def cal_loss(self, input, labels, model, target_index=[0], log=True):
        return cal_loss(input, labels, model, self.device, target_index, log)
    
    def cal_loss_batch(self, input, model, target_batch_index=[0], target_index=[0], log=True):
        return cal_loss_batch(input, model, self.device, target_batch_index, target_index, log)
    
    def cal_MSE(self, input, target):
        return cal_MSE(input, target)
    
    def cal_PSNR(self, input, target):
        return cal_PSNR(input, target)
    
    def cal_SSIM(self, input, target, window_size=11, window=None, size_average=True, full=False, val_range=None):
        return cal_SSIM(input, target, window_size, window, size_average, full, val_range)
    
    def cal_LPIPS(self, input, target):
        return cal_LPIPS(input, target, self.lp_model)
    


#Logger    
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
      "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger



    
