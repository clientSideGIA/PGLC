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


"""
Other Inversion Attack
- DLG
- iDLG
- GradInversion
- DeepInversion
"""

"""
DeepInversion
[1] H. Yin et al., ‘Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion’, presented at the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 8715–8724.
Ref: https://github.com/NVlabs/DeepInversion 
"""

def total_variation(x):
  
    diff1=torch.norm(x[:,:,:,:-1] - x[:,:,:,1:])
    diff2=torch.norm(x[:,:,:-1,:] - x[:,:,1:,:])
   
    return diff1+diff2


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

class DeepInversionFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def feature_loss(dummy_data,class_mean,class_var):
    nch = dummy_data.shape[1]
    mean = dummy_data.mean([0, 2, 3])
    var = dummy_data.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

    r_feature = torch.norm(class_var - var, 2) + torch.norm(
            class_mean - mean, 2)

    return r_feature


def deep_inversion(model,batch_size,target_class,logger,save_name,
                   epoch_num=20000,
                   main_coeff=1e-2,
                   l2_coeff=1e-5,
                   tv_coeff=1e-4,
                   bn_coeff=1e-2,
                   first_bn_weight=10,
                   lr=0.25,
                   image_size=64,
                   save_path="./temp_data/",
                   ifprint=True,
                   ifhistory=False):


    random.seed(0)
    torch.manual_seed(torch.cuda.current_device())

    model.eval()
    loss_r_feature_layers = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))


    target_labels = torch.LongTensor([target_class]*batch_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    dummy_data = torch.randn([batch_size,3,image_size,image_size]).to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([dummy_data],lr=lr,betas=[0.5, 0.9], eps = 1e-8)

    lr_scheduler = lr_cosine_policy(lr, 100, epoch_num)

    history = []
    best_cost = 1e6
    best_output = None
    
    for t in range(epoch_num):
        lr_scheduler(optimizer, t, t)

        optimizer.zero_grad()
        model.zero_grad()

        dummy_pred = model(dummy_data) 
        main_loss = criterion(dummy_pred, target_labels)

        loss = main_coeff * main_loss
        loss = loss + l2_coeff * torch.norm(dummy_data.view(batch_size, -1), dim=1).mean()
        loss = loss + tv_coeff * total_variation(dummy_data)

        bn_weights = [first_bn_weight]+[1. for _ in range(len(loss_r_feature_layers)-1)]
        bn_loss = sum([module.r_feature * bn_weights[idx] for idx,module in enumerate(loss_r_feature_layers)])
        loss = loss + bn_coeff * bn_loss

        loss.backward()
        optimizer.step()

        if best_cost > loss.item() or t == 0:
            best_dummy_data = dummy_data.data.clone()
            best_cost = loss.item()


        if (t+1) % 1000 == 0: 
            current_loss = loss
            if ifprint:
                print("---------------Iter %d--------------" % t)
                print("Total loss, %.8f" % current_loss.item())
                print("R_feature loss, %.8f" % bn_loss.item())
                print("Main criterion loss, %.8f" % main_loss.item())

            # vutils.save_image(dummy_data,'/content/{}.png'.format(t),
            #                     normalize=True, 
            #                     scale_each=True, 
            #                     nrow=int(10))
            if ifhistory:
                history.append(dummy_data.cpu())
    
    history.append(dummy_data.cpu())
    vutils.save_image(dummy_data,save_path+"{}_image.png".format(save_name),
                            normalize=False, 
                            scale_each=True, 
                            nrow=int(10))

    best_output = best_dummy_data
    torch.save(best_output,save_path+save_name+".tensor")

    return history,best_output