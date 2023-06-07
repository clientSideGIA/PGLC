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

from gradient_lib import *
from utils import *

'''
Deep Learning Lib
-Epoch Lib
'''

def epoch(loader, model, opt, rounds=None):
    '''
    For normal training.
    '''
    model.train()
    total_loss, total_acc = 0., 0.
    curr_round = 0

    if rounds == None:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]

        return total_acc / len(loader.dataset), total_loss / len(loader.dataset)

    else:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]
            curr_round += 1
            if curr_round == rounds:
                break

        return total_acc / (X.shape[0] * rounds), total_loss / (X.shape[0] * rounds)

def epoch_test(loader, model):
    model.eval()
    total_loss, total_acc = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_target(loader, model, target_class):
    model.eval()

    target_num = 0.
    target_loss = 0.
    correct_target_num = 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if len(y[y == target_class]) == 0:
            continue

        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        target_num += len(y[y == target_class])
        pred_label = yp.max(dim=1)[1]
        correct_target_num += (pred_label[y == target_class]
                               == target_class).sum().item()

    return correct_target_num/target_num


def epoch_target2(loader, model, target_class):
    model.eval()

    target_num = 0.
    total_acc = 0.
    total_loss = 0.

    for X, y in loader:
        X = X[y == target_class]
        y = y[y == target_class]

        X, y = X.to(device), y.to(device)
        if len(y) == 0:
            continue
        else:
            target_num += len(y)

        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_acc / target_num, total_loss / target_num


def epoch_FP(loader, model, target_class):
    model.eval()

    total_num = 0.
    fp_num = 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        total_num += len(y[y != target_class])
        pred_label = yp.max(dim=1)[1]
        fp_num += (pred_label[y != target_class] == target_class).sum().item()

    return fp_num/total_num


def epoch_eval_gradient(loader, model):
    model.eval()
    gradient_list = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        gradient = torch.autograd.grad(loss, model.parameters())
        gradient = [grad.detach() for grad in gradient]
        gradient_list.append(gradient)

    avg_gradient = cal_gradient_mean(gradient_list)
    return avg_gradient

def label_to_onehot(target, num_classes):
    if type(target) is list:
        target = torch.tensor(target)

    if len(target.shape) == 0:
        target = target.reshape([1])

    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)

    return onehot_target
