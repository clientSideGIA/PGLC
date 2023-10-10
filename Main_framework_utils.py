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

from dataset import *
from common_DL import *
from gradient_lib import *
from federated_learning import *
from model_structure import *
from utils import *
from inversion_attacks import *
from model_structure import *




"""
 Utils
- TargetBatchGenerator
- Sampling Methods for Datasets
- Metrics
- FedSGD Tools
"""


class TargetBatchGenerator:
    """
    Generator class used to generate a batch including target samples
    """
    def __init__(self, dataset, target_class=0, batch_size=16, target_class_num=1, target_fixed=False):
        images = np.array(dataset.images)
        labels = np.array(dataset.labels)
        self.images = images
        self.labels = labels

        self.dataset = dataset
        self.target_class = target_class
        self.batch_size = batch_size
        self.target_class_num = target_class_num
        self.target_images = images[labels == target_class]
        self.target_index = np.argwhere(
            labels == target_class).reshape(1, -1)[0]
        self.general_images = images[labels != target_class]
        self.general_labels = labels[labels != target_class]
        self.general_index = np.argwhere(
            labels != target_class).reshape(1, -1)[0]

        self.target_fixed = target_fixed

    def fixed_target(self, fixed_target_images):
        self.fixed_target_images = fixed_target_images

    def choose_target_batch_data(self):
        index = []

        batch_size = self.batch_size
        target_class_num = self.target_class_num

        index += np.random.choice(self.target_index, size=target_class_num).tolist()
        index += np.random.choice(self.general_index,
                                  size=batch_size-target_class_num).tolist()

        return index

    def generate_target_batch_data(self):
        index = self.choose_target_batch_data()
        images = []
        labels = []

        for i in index:
            sample, label = self.dataset[i]
            images.append(sample)
            labels.append(label)

        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)

        return images, labels

    def generate_fixed_target_batch_data(self):
        index = self.choose_target_batch_data()

        images = []
        labels = []

        for i in index:
            sample, label = self.dataset[i]
            images.append(sample)
            labels.append(label)

        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)

        images = images[self.target_class_num:]
        images = torch.concat((self.fixed_target_images, images), dim=0)

        return images, labels

    def __iter__(self):
        return self

    def __next__(self):
        if not self.target_fixed:
            X, y = self.generate_target_batch_data()
        else:
            X, y = self.generate_fixed_target_batch_data()
        return X, y
    
    
    
"""
Sampling Methods Lib

"""

def sampling_datasets_iid(dataset,dataset_num,each_dataset_size=None,dcopy=False,transform=None):
    datasets = []
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    image_label_dict = {}
    
    if each_dataset_size == None:
        each_dataset_size = len(dataset)
    
    indices = list(range(len(labels)))
    
    for i in range(dataset_num):
        if dcopy:
            new_dataset = copy.deepcopy(dataset)
        else:
            new_dataset = copy.copy(dataset)

        new_dataset.images = []
        new_dataset.labels = []

        if transform!=None:
            new_dataset.transform = transform

        new_datset_index = np.random.choice(indices,size=each_dataset_size)
        new_dataset.images = images[new_datset_index]
        new_dataset.labels = labels[new_datset_index]
        

        datasets.append(new_dataset)
    
    return datasets

def sampling_datasets(dataset,dataset_num,label_range,each_class_num,dcopy=False,transform=None,label_index=None):
    datasets = []
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    image_label_dict = {}

    for label in range(0,label_range):
        image_label_dict[label] = images[labels==label]

    for i in range(dataset_num):
        if dcopy:
            new_dataset = copy.deepcopy(dataset)
        else:
            new_dataset = copy.copy(dataset)
        new_dataset.images = []
        new_dataset.labels = []
        if transform!=None:
            new_dataset.transform = transform
        if label_index == None:
            for label in range(0,label_range):
                new_dataset.images += (np.random.choice(image_label_dict[label],size=each_class_num).tolist())
                new_dataset.labels += [label]*each_class_num
        else:
            for label in label_index:
                new_dataset.images += (np.random.choice(image_label_dict[label],size=each_class_num).tolist())
                new_dataset.labels += [label]*each_class_num

        datasets.append(new_dataset)

    return datasets

def sampling_dataset(dataset,label_range,each_class_num,dcopy=False,transform=None,label_index=None):
    datasets = []
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    image_label_dict = {}

    for label in range(0,label_range):
        image_label_dict[label] = images[labels==label]

   
    if dcopy:
        new_dataset = copy.deepcopy(dataset)
    else:
        new_dataset = copy.copy(dataset)
    new_dataset.images = []
    new_dataset.labels = []
    if transform!=None:
        new_dataset.transform = transform
    if label_index == None:
        for label in range(0,label_range):
            new_dataset.images += (np.random.choice(image_label_dict[label],size=each_class_num).tolist())
            new_dataset.labels += [label]*each_class_num
    else:
        for label in label_index:
            new_dataset.images += (np.random.choice(image_label_dict[label],size=each_class_num).tolist())
            new_dataset.labels += [label]*each_class_num

        

    return new_dataset


def sampling_only_target_class_dataset(dataset, target_class, target_class_num, label_range, dcopy=False, transform=None):
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    image_label_dict = {}

    for label in range(0, label_range):
        image_label_dict[label] = images[labels == label]

    if dcopy:
        new_dataset = copy.deepcopy(dataset)
    else:
        new_dataset = copy.copy(dataset)

    new_dataset.images = []
    new_dataset.labels = []
    
    if transform != None:
        new_dataset.transform = transform

    new_dataset.images += (np.random.choice(
        image_label_dict[target_class], size=target_class_num).tolist())
    new_dataset.labels += [target_class]*target_class_num

    return new_dataset


def sampling_with_target_class_dataset(dataset, target_class, target_class_num, label_range, each_class_num, dcopy=False, transform=None):
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    image_label_dict = {}

    for label in range(0, label_range):
        image_label_dict[label] = images[labels == label]

    if dcopy:
        new_dataset = copy.deepcopy(dataset)
    else:
        new_dataset = copy.copy(dataset)

    new_dataset.images = []
    new_dataset.labels = []
    if transform != None:
        new_dataset.transform = transform

    for label in range(0, label_range):
        if label == target_class:
            new_dataset.images += (np.random.choice(
                image_label_dict[target_class], size=target_class_num).tolist())
            new_dataset.labels += [target_class]*target_class_num

        else:
            new_dataset.images += (np.random.choice(
                image_label_dict[label], size=each_class_num).tolist())
            new_dataset.labels += [label]*each_class_num

    return new_dataset


def sampling_no_target_class_dataset(dataset, target_class, dataset_size=100000, dcopy=False, transform=None):
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)

    if dcopy:
        new_dataset = copy.deepcopy(dataset)
    else:
        new_dataset = copy.copy(dataset)

    new_dataset.images = []
    new_dataset.labels = []
    if transform != None:
        new_dataset.transform = transform

    general_images = images[labels != target_class]
    general_labels = labels[labels != target_class]
    general_size = general_labels.shape[0]

    index = np.random.choice(range(general_size), size=dataset_size)

    new_dataset.images += general_images[index].tolist()
    new_dataset.labels += general_labels[index].tolist()

    return new_dataset


def sampling_no_target_class_datasets(dataset, dataset_num, target_class, dataset_size=100000, dcopy=False, transform=None):
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)

    general_images = images[labels != target_class]
    general_labels = labels[labels != target_class]
    general_size = general_labels.shape[0]

    no_target_class_datasets = []

    for _ in range(dataset_num):

        if dcopy:
            new_dataset = copy.deepcopy(dataset)
        else:
            new_dataset = copy.copy(dataset)

        new_dataset.images = []
        new_dataset.labels = []

        if transform != None:
            new_dataset.transform = transform

        index = np.random.choice(range(general_size), size=dataset_size)

        new_dataset.images += general_images[index].tolist()
        new_dataset.labels += general_labels[index].tolist()

        no_target_class_datasets.append(new_dataset)

    return no_target_class_datasets

def remove_target_class_dataset(dataset, target_class, dcopy=False, transform=None):
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)

    general_images = images[labels != target_class]
    general_labels = labels[labels != target_class]
    general_size = general_labels.shape[0]

    if dcopy:
        new_dataset = copy.deepcopy(dataset)
    else:
        new_dataset = copy.copy(dataset)

    new_dataset.images = general_images.tolist()
    new_dataset.labels = general_labels.tolist()

    if transform != None:
        new_dataset.transform = transform

    return new_dataset


def sampling_no_target_class_dataset_niid(dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None):
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    image_label_dict = {}

    for label in range(0, label_range):
        image_label_dict[label] = images[labels == label]

    if dcopy:
        new_dataset = copy.deepcopy(dataset)
    else:
        new_dataset = copy.copy(dataset)

    new_dataset.images = []
    new_dataset.labels = []

    if transform != None:
        new_dataset.transform = transform

    for label in range(0, label_range):
        if label == target_class:
            continue
        elif label in label_list:
            new_dataset.images += (np.random.choice(
                image_label_dict[label], size=each_class_num).tolist())
            new_dataset.labels += [label]*each_class_num

    return new_dataset


"""
Metrics
"""


def cal_loss_gap_based_on_batch(input, model, target_batch_index=[0], target_index=[0], ifprint=True):
    """
    Calculate Loss Gap (How much the targeted gradient dominates the aggregation)
    """
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
            target_images, target_labels = target_images.to(device), target_labels.to(device)
            for no_target_idx in no_target_index:
                yp = model(target_images[no_target_idx:no_target_idx+1])
                no_tagret_loss = criterion(
                    yp, target_labels[no_target_idx:no_target_idx+1])
                total_other_loss += no_tagret_loss.item()

        for batch_idx in range(len(input)):
            if batch_idx not in target_batch_index:
                images, labels = input[batch_idx]
                images, labels = images.to(device), labels.to(device)
                yp = model(images)
                loss = criterion(yp, labels)
                total_other_loss += loss.item()*len(images)

        total_loss = 0.
        for batch_idx in range(len(input)):
            images, labels = input[batch_idx]
            images, labels = images.to(device), labels.to(device)
            yp = model(images)
            loss = criterion(yp, labels)
            total_loss += loss.item()

        if ifprint:
            print("Target Loss:", total_target_loss)
            print("Other Loss:", total_other_loss)
            print("Loss Gap:", total_target_loss/(total_target_loss+total_other_loss))
            print("Total Loss:", total_loss/len(input))

        return total_target_loss, total_other_loss, total_loss/len(input), total_target_loss/(total_target_loss+total_other_loss)
    
def eval_image(dummy_data,original_input):
    size = original_input.shape[0]
    dummy_data = dummy_data.to(device)
    original_input = original_input.to(device)
    image_metric_list = []
    for i in range(size):
        PSNR_val = cal_PSNR(dummy_data[i:i+1],original_input[i:i+1])
        LPIPS_val = cal_LPIPS(dummy_data[i:i+1],original_input[i:i+1])
        SSIM_val = cal_SSIM(dummy_data[i:i+1],original_input[i:i+1])
        
        image_metric_list.append({
            "PSNR":PSNR_val,
            "LPIPS":LPIPS_val,
            "SSIM":SSIM_val
        })
        
        print("-----Target %d-----" % i)
        print("PSNR: %.8f" % PSNR_val)
        print("LPIPS: %.8f" % LPIPS_val)
        print("SSIM: %.8f" % SSIM_val)
    
    return image_metric_list



"""
 FedSGD Utils
"""


# Batch Inference Under AGR
def fedSGD_batch_gradient_inference_under_AGR(server, clients, input, return_candidx=True):
    client_num = len(clients)

    gradient_list = []
    bn_data_list = []

    state_dict_list = []

    
    AGR = server.aggregate_grads

    for client in clients:
        client.set_server(server)

    last_global_weights = server.get_weights()
    
    previous_global_model_state_dict = copy.deepcopy(server.global_model.state_dict())

    for client, input_tuple in zip(clients, input):
        client.client_model.eval()
        X, y = input_tuple

        gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(X, y)

        gradient_list.append(gradient)
        bn_data_list.append(bn_data)

    
    aggregation, candidate_indices = AGR(gradient_list, server.n_attacker, return_candidx=True)
    server.train_sgd_batch(gradient_list, bn_data_list)
    current_global_weights = server.get_weights()


    inference_gradient = infer_global_aggregated_gradient(
            last_global_weights, current_global_weights, scale=100)
    
    lr = server.get_lr()
    print("#Client:", client_num)
    print("Lr:", lr)

    if return_candidx:
        return gradient_list, inference_gradient, previous_global_model_state_dict, candidate_indices
    else:
        return gradient_list, inference_gradient, previous_global_model_state_dict
    

def infer_global_aggregated_gradient(last_global_weights, current_global_weights, scale=100):
    inference_gradient = [(before_p.detach() - after_p.detach())*torch.tensor(scale)
                               for before_p, after_p in zip(last_global_weights, current_global_weights)]

    return inference_gradient

def calculate_replacement_gradient(current_model,replace_model_state_dict, scale=100):

    model = current_model
    current_model_state_dict = deepcopy(model.state_dict())
    #current_model_state_dict = model.state_dict()
    current_model_weights = [p.detach().clone()
                             for p in model.parameters()]

    model.load_state_dict(replace_model_state_dict)
    replace_model_weights = [p.detach().clone()
                             for p in model.parameters()]

    replacement_gradient = [scale * (current_model_weight - replace_model_weight) for current_model_weight,
                replace_model_weight in zip(current_model_weights, replace_model_weights)]

    model.load_state_dict(current_model_state_dict)

    return replacement_gradient
    

def cal_model_distance(model1,model2):

    normal_parameters =  [p.detach().clone()
                                 for p in model1.parameters()]
    target_parameters =  [p.detach().clone()
                                 for p in model2.parameters()]
    
    model_distance = cal_gradient_distance_by_dimension(normal_parameters,target_parameters)
    
    return model_distance

def cal_gradient_distance_by_dimension(gradient1, gradient2):
    distance = 0.
    for grad1, grad2 in zip(gradient1, gradient2):
        distance += (grad1 - grad2).sum()**2/grad1.reshape(1,-1).shape[1]
    
    distance = distance.sqrt().item()
    return distance