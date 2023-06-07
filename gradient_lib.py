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
Gradient Lib

'''

def cal_gradient(model, original_input, original_label):
    '''
    Directly calculate gradient

    '''
    model = model.to(device)
    original_input = original_input.to(device)
    logits = model(original_input)
    #original_onehot_label = label_to_onehot(original_label,num_classes).to(device)
    label = original_label.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    pred = model(original_input)
    loss = criterion(pred, label)
    print("Loss: ", loss)
    gradients = torch.autograd.grad(loss, model.parameters())

    return gradients


def record_intermediate_output(model, module_class=nn.ReLU):
    capture_layers = []
    for module in model.modules():
        if isinstance(module, module_class):
            capture_layers.append(IMCapture(module))
    return capture_layers


def show_gradient(model, gradients, conv_bias=False, bn_aff=True):
    '''
    Visualize gradients

    '''
    gradients = [grad for grad in gradients]
    weight_grad = []
    index = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            grad = gradients[index].reshape(-1)
            zero_p = (grad == 0.).sum()/grad.shape[0]
            weight_grad.append(("Conv", grad, zero_p))
            if conv_bias:
                index += 2
            else:
                index += 1

        if isinstance(module, nn.Linear):
            grad = gradients[index].reshape(-1)
            zero_p = (grad == 0.).sum()/grad.shape[0]
            weight_grad.append(("Linear", grad, zero_p))
            index += 2

        if isinstance(module, nn.BatchNorm2d):
            if bn_aff:
                index += 2

    conv_index = 0
    linear_index = 0

    weight_size = len(weight_grad)
    col_size = 4
    row_size = math.ceil(weight_size/col_size)

    fig, ax = plt.subplots(row_size, col_size, figsize=(
        12, 8), constrained_layout=True)
    for index, item in enumerate(weight_grad):
        type_name, grad, zero_p = item
        if type_name == "Conv":
            subfig_name = "Conv %d" % conv_index
            conv_index += 1
        elif type_name == "Linear":
            subfig_name = "Linear %d" % linear_index
            linear_index += 1

        title = (subfig_name + " zero_p: %.4f") % zero_p
        x_axis = np.linspace(torch.min(grad).item(),
                             torch.max(grad).item(), num=100)
        #x_axis = np.linspace(-10, 10, num=100)
        y_axis = torch.histc(grad, bins=100).cpu().numpy()
        ax[index//col_size][index % col_size].plot(x_axis, y_axis)
        ax[index//col_size][index % col_size].set_title(title)

        print(subfig_name, "-Norm2:", torch.linalg.norm(grad, 2))

    fig.show()
    return weight_grad


def show_intermediate_output(capture_layers, module_name):
    layer_size = len(capture_layers)
    col_size = 4
    row_size = math.ceil(layer_size/col_size)

    fig, ax = plt.subplots(row_size, col_size, figsize=(
        12, 8), constrained_layout=True)
    for index, capture in enumerate(capture_layers):
        subfig_name = (module_name + " %d ") % index
        inter_data = capture.output.reshape(-1)
        zero_p = (inter_data == 0.0).sum()/inter_data.shape[0]
        title = (subfig_name + "Output zero_p: %.4f") % zero_p

        x_axis = np.linspace(torch.min(inter_data).item(),
                             torch.max(inter_data).item(), num=100)
        #x_axis = np.linspace(-10, 10, num=100)
        y_axis = torch.histc(inter_data, bins=100).cpu().numpy()
        ax[index//col_size][index % col_size].plot(x_axis, y_axis)
        ax[index//col_size][index % col_size].set_title(title)

        print(subfig_name, "-Norm2:", torch.linalg.norm(inter_data, 2))

    fig.show()


def cal_gradient_norm(gradient, norm=2):
    norm_sum = 0.
    for grad in gradient:
        norm_sum += torch.linalg.norm(grad.reshape(-1), norm)

    return norm_sum


def cal_gradient_var_based_layer(gradient):
    var_list = []
    for grad in gradient:
        var_list.append(torch.var(grad.reshape(-1)))

    return var_list


def cal_gradient_norm_based_layer(gradient, norm=2):
    norm_list = []
    for grad in gradient:
        norm_list.append(torch.linalg.norm(grad.reshape(-1), norm))

    return norm_list


def cal_layer_gradient_norm(gradient, norm=2):
    norm_list = []
    for grad in gradient:
        norm_list.append(torch.linalg.norm(grad.reshape(-1), norm))

    return norm_list


def show_layer_gradient_norm(norm_list):
    for i in range(0, len(norm_list), 2):
        print("Layer %d weight:" % i, norm_list[i])
        print("Layer %d bias:" % i, norm_list[i+1])


def show_layer_gradient_var(var_list):
    for i in range(0, len(var_list), 2):
        print("Layer %d weight:" % i, var_list[i])
        print("Layer %d bias:" % i, var_list[i+1])


def cal_gradient_distance_sum(gradient_list):
    distance_list = []
    for current_grad in gradient_list:
        distance = 0.
        for grad in gradient_list:
            distance += cal_gradient_distance(current_grad, grad)
        distance_list.append(distance)
    return distance_list

def cal_gradient_distance_sum2(distance_array,indices):
    indices = np.array(indices)
    distance_array = np.array(distance_array)[indices][:,indices]
    distance_list = np.sum(distance_array,axis=1)

    return distance_list

def cal_gradient_distance_array(gradient_list):
    distance_list = []
    for i, current_grad in enumerate(gradient_list):
        distances = []
        for j, grad in enumerate(gradient_list):
            if i == j:
                continue
            distances.append(cal_gradient_distance(current_grad, grad))
        distance_list.append(distances)
    return distance_list

def cal_gradient_distance_array2(gradient_list):
    distance_list = []
    for i,current_grad in enumerate(gradient_list):
        distances = []
        for j,grad in enumerate(gradient_list):
            distances.append(cal_gradient_distance(current_grad,grad))
        distance_list.append(distances)
    return distance_list

def cal_gradient_distance(gradient1, gradient2):
    distance = 0.
    for grad1, grad2 in zip(gradient1, gradient2):
        distance += torch.linalg.norm(grad1 - grad2).item()

    return distance


def cal_gradient_cos_similarity(gradient1, gradient2):
    costs = 0.
    pnorm = [0., 0.]
    for grad1, grad2 in zip(gradient1, gradient2):
        costs += (grad1 * grad2).sum()
        pnorm[0] += grad1.pow(2).sum()
        pnorm[1] += grad2.pow(2).sum()

    similarity = costs / pnorm[0].sqrt() / pnorm[1].sqrt()

    return similarity

def cal_gradient_if_zero(gradient):
    cost = 0.
    for grad in gradient:
        cost += grad.sum()
    
    return cost == 0.


def cal_gradient_mean(gradient_list):
    size = len(gradient_list[0])
    gradients = {index: [] for index in range(0, size)}

    for gradient in gradient_list:
        for index, grad in enumerate(gradient):
            gradients[index].append(grad)

    avg_gradients = []
    for index in gradients.keys():
        avg_gradient_layer = torch.mean(torch.stack(gradients[index]), dim=0)
        avg_gradients.append(avg_gradient_layer)

    return avg_gradients


def convert_gradient_list_to_layer_list(gradient_list):
    layer_gradient_list = [[] for _ in range(len(gradient_list[0]))]

    for gradient in gradient_list:

        for i, grad in enumerate(gradient):
            layer_gradient_list[i].append(grad)

    return layer_gradient_list


def cal_gradient_median_based_layer(layer_gradient_list, return_idx=False):
    gradient = []
    candidate_idx = []
    for layer in layer_gradient_list:
        value, idx = torch.median(torch.stack(layer), dim=0)
        gradient.append(value)
        candidate_idx.append(idx)

    if return_idx:
        return gradient, candidate_idx
    return gradient


def cal_gradient_distance_list(gradient_list, current_gradient):
    distance = []
    for gradient in gradient_list:
        dis = 0.
        for grad, current_grad in zip(gradient, current_gradient):
            dis += torch.linalg.norm(grad - current_grad).item()

        distance.append(dis)

    return distance

# def cal_gradient_distance_based_layer(layer_gradient_list,current_gradient):
#   distance = [[] for _ in range(len(current_gradient))]

#   for i_layer_grad_list,current_layer_grad in zip(enumerate(layer_gradient_list),current_gradient):
#     i,layer_grad_list = i_layer_grad_list
#     for layer_grad in layer_grad_list:
#       print(torch.linalg.norm(layer_grad - current_layer_grad).item())
#       raise
#       distance[i].append(torch.linalg.norm(layer_grad - current_layer_grad).item())

#   return distance


def cal_gradient_mean_based_layer(layer_gradient_list):
    gradient = []
    for layer in layer_gradient_list:
        gradient.append(torch.mean(torch.stack(layer)[0], dim=0))

    return gradient

def accumulative_mean_grad(last_mean_gradient, new_gradient, momentum=0.8):
    momentum = torch.tensor(momentum).to(device)
    new_mean_gradient = []
    for last_mean_grad, new_grad in zip(last_mean_gradient, new_gradient):
        new_mean_gradient.append(
            last_mean_grad * momentum + new_grad * (1-momentum))

    return new_mean_gradient

def cal_gradient_unit(sample_gradient):
    unit_gradient_list = []
    norm_sum = 0.
    for sample_grad in sample_gradient:
        norm_sum += sample_grad.pow(2).sum()

    norm = norm_sum.sqrt()

    for sample_grad in sample_gradient:
        unit_grad = sample_grad / norm
        unit_gradient_list.append(unit_grad)

    return unit_gradient_list


def gradient_cos_similarity_loss(current_gradient, target_gradient):
    costs = 0.
    pnorm = [0., 0.]
    for current_grad, target_grad in zip(current_gradient, target_gradient):
        costs -= (current_grad * target_grad).sum()
        pnorm[0] += current_grad.pow(2).sum()
        pnorm[1] += target_grad.pow(2).sum()

    similarity_cost = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

    return similarity_cost


def gradient_distance_loss(current_gradient, target_gradient):
    distance_sum = 0
    for current_grad, target_grad in zip(current_gradient, target_gradient):
        distance_sum += ((current_grad - target_grad) ** 2).sum()

    return distance_sum


def gradient_norm_loss(current_gradient, target_gradient):
    gradient_norm = torch.sqrt((cal_gradient_norm(
        current_gradient) - cal_gradient_norm(target_gradient)) ** 2)
    return gradient_norm


def init_gradient(sample_gradient):
    init_gradient_list = []
    for sample_grad in sample_gradient:
        init_grad = torch.randn_like(sample_grad).to(
            device).requires_grad_(True)
        init_gradient_list.append(init_grad)

    return init_gradient_list


def copy_gradient(sample_gradient):
    init_gradient_list = []
    for sample_grad in sample_gradient:
        init_grad = torch.randn_like(sample_grad).copy_(sample_grad)
        init_grad = init_grad.to(device).requires_grad_(True)
        init_gradient_list.append(init_grad)

    return init_gradient_list


def copy_gradient_sign(sample_gradient):
    init_gradient_list = []
    for sample_grad in sample_gradient:
        init_grad = torch.randn_like(sample_grad).copy_(sample_grad).sign()
        init_grad = init_grad.to(device).requires_grad_(True)
        init_gradient_list.append(init_grad)

    return init_gradient_list


def copy_gradient_unit(sample_gradient):
    init_gradient_list = []
    norm_sum = 0.
    for sample_grad in sample_gradient:
        norm_sum += sample_grad.pow(2).sum()

    norm = norm_sum.sqrt()

    for sample_grad in sample_gradient:
        init_grad = torch.randn_like(sample_grad).copy_(sample_grad)
        init_grad = init_grad / norm
        init_grad = init_grad.to(device).requires_grad_(True)
        init_gradient_list.append(init_grad)

    return init_gradient_list