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

from dataset import *
from common_DL import *
from gradient_lib import *
from federated_learning import *
from model_structure import *
from utils import *
from inversion_attacks import *
from CGI_framework_utils import *

import  global_var 
device = global_var.get_device()

def epoch_class_trap(loader, model, opt, target_class):
    '''
    For trap training (full knowledge).
    '''
    model.train()
    total_loss, total_acc = 0., 0.
    for X, y in loader:
        y = torch.tensor([target_class]*len(y))
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = -nn.CrossEntropyLoss()(yp, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_poisoning(loader, model, opt):
    '''
    For poisoning training (semi knowledge).
    '''
    model.train()
    total_loss, total_acc = 0., 0.
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = -nn.CrossEntropyLoss()(yp, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)


def tensor2loader(tensor_data,target_class,batch_size,loader_size):
    tensor_dataset_X = []
    tensor_dataset_y = []
    
    for _ in range(loader_size):
        X = tensor_data[torch.randint(len(tensor_data), (batch_size,))]
        y = torch.tensor([target_class]*batch_size,dtype=torch.long).to(device)
        
        tensor_dataset_X.append(X)
        tensor_dataset_y.append(y)
        
    tensor_dataset_X = torch.cat(tensor_dataset_X,dim=0) 
    tensor_dataset_y = torch.cat(tensor_dataset_y,dim=0) 
    
    tensor_dataset = torch.utils.data.TensorDataset(tensor_dataset_X, tensor_dataset_y)
    
    loader = torch.utils.data.DataLoader(tensor_dataset,batch_size=batch_size,shuffle=False)
    
    return loader

def epoch_model_inversion_poisoning(loader, model, opt):
    '''
    For model inversion poisoning training (no knowledge).
    '''
    model.train()
    total_loss, total_acc = 0., 0.
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = -nn.CrossEntropyLoss()(yp, y)

        model.train()
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)

# hyperparameter_dict = {
#                 "main_task_lr" : 1e-2,
#                 "main_task_lr_decay" : True,
#                 "main_task_lr_decay_gamma" : 0.1,
#                 "main_task_lr_decay_milestone" : [15],
#                 "malicious_task_lr" : 2e-5,
#                 "save_interal" : None,
#                 "epoch_num" : 30
#             }

def training_malicious_model_with_full_knowledge(dataset_name, client, target_class, no_target_train_loader, maintask_loader, entire_train_loader, hyperparameter_dict, logger, 
                                                 save_path="./model/", save_name=None, ifprint=True):
  

    hydict = hyperparameter_dict
    suffix = ".pth"
    if save_name == None:
        save_name = dataset_name+"_FK_"+str(target_class)+"_Malicious_Model"
        
    model = client.client_model
    main_task_opt = optim.SGD(model.parameters(), lr=hydict["main_task_lr"],momentum=0.8)
    malicious_task_opt = optim.SGD(model.parameters(), lr=hydict["malicious_task_lr"])
    
    if hydict["main_task_lr_decay"]:
        main_task_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(main_task_opt, 
                                                                       milestones=hydict["main_task_lr_decay_milestone"], 
                                                                            gamma=hydict["main_task_lr_decay_gamma"])
    
    try:
        for t in range(hydict["epoch_num"]):
            begin = time.time()
            
            train_acc, train_loss = epoch(no_target_train_loader, model, main_task_opt)
            _, trap_loss = epoch_class_trap(no_target_train_loader, model, malicious_task_opt,target_class)
            
            maintask_acc, maintask_loss = epoch_test(maintask_loader,model)
            target_acc, target_loss = epoch_target2(entire_train_loader, model, target_class)
            
            if hydict["main_task_lr_decay"]:
                main_task_opt_scheduler.step()
            
            end = time.time()
            time_elapsed = end - begin
            
            if ifprint:
                print("-------------{} FK Epoch: {:d}--------------" .format(dataset_name, t))
                print("Model Train Loss: {:.6f}".format(train_loss))
                print("Model Trap Loss: {:.6f}".format(trap_loss))
                print("Model Accuracy on No Target Train Dataset: {:.6f}".format(train_acc))
                print("Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(maintask_acc, maintask_loss))
                print("Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(target_acc, target_loss))
                print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            
            logger.info("-------------{} FK Epoch: {:d}--------------" .format(dataset_name, t))
            logger.info("Model Train Loss: {:.6f}".format(train_loss))
            logger.info("Model Trap Loss: {:.6f}".format(trap_loss))
            logger.info("Model Accuracy on No Target Train Dataset: {:.6f}".format(train_acc))
            logger.info("Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(maintask_acc, maintask_loss))
            logger.info("Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(target_acc, target_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            
            if hydict["save_interal"] != None:
                if t % hydict["save_interal"]==0:
                    client.save_model(path+save_name+"_"+str(t)+suffix)
            
        client.save_model(save_path+save_name+suffix)
        
    except KeyboardInterrupt:
        print("Stopping")
    
     
    
     
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-4,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 5e-3,
#                 "save_interal" : None,
#                 "epoch_num" : 3
#             }

def training_malicious_model_with_semi_knowledge(dataset_name, client, target_class, only_target_train_loader, local_train_loader, main_task_loader, entire_train_loader,                                                                    hyperparameter_dict, logger, save_path="./model/", save_name=None, ifprint=True):
  

    
    hydict = hyperparameter_dict
    suffix = ".pth"
    if save_name == None:
        save_name = dataset_name+"_SK_"+str(target_class)+"_Malicious_Model"
        
        
    model = client.client_model
    main_task_opt = optim.SGD(model.parameters(), lr=hydict["main_task_lr"])
    malicious_task_opt= optim.SGD(model.parameters(), lr=hydict["malicious_task_lr"], momentum = 0.9)
    
    
    if hydict["main_task_lr_decay"]:
        main_task_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(main_task_opt, 
                                                                       milestones=hydict["main_task_lr_decay_milestone"], 
                                                                            gamma=hydict["main_task_lr_decay_gamma"])
    
    try:
        for t in range(hydict["epoch_num"]):
            begin = time.time()
            
            _, _ = epoch_poisoning(only_target_train_loader, model, malicious_task_opt)
            train_acc, train_loss = epoch(local_train_loader, model, main_task_opt)
            
            maintask_acc, maintask_loss = epoch_test(main_task_loader, model)
            target_acc, target_loss = epoch_target2(entire_train_loader, model, target_class)

            if hydict["main_task_lr_decay"]:
                main_task_opt_scheduler.step()
            
            end = time.time()
            time_elapsed = end - begin
            
            if ifprint:
                print("-------------{} SK Epoch: {:d}--------------".format(dataset_name, t))
                print("Model Train Loss: {:.6f}".format(train_loss))
                print("Model Accuracy on Local Train Dataset: {:.6f}".format(train_acc))
                print("Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(maintask_acc, maintask_loss))
                print("Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(target_acc, target_loss))
                print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            
            logger.info("-------------{} SK Epoch: {:d}--------------".format(dataset_name, t))
            logger.info("Model Train Loss: {:.6f}".format(train_loss))
            logger.info("Model Accuracy on Local Train Dataset: {:.6f}".format(train_acc))
            logger.info("Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(maintask_acc, maintask_loss))
            logger.info("Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(target_acc, target_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            
            if hydict["save_interal"] != None:
                if t % hydict["save_interal"]==0:
                    client.save_model(save_path+save_name+"_"+str(t)+suffix)
            
        client.save_model(save_path+save_name+suffix)
        
    except KeyboardInterrupt:
        print("Stopping")      



        
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-5,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 1e-3,
#                 "save_interal" : None,
#                 "epoch_num" : 2
#             }

def training_malicious_model_with_no_knowledge(dataset_name, client, target_class, only_inversion_target_Cifar100_train_loader, local_train_loader, main_task_loader, entire_train_loader,                                                                    hyperparameter_dict, logger, save_path="./model/", save_name=None, ifprint=True):
  

    
    hydict = hyperparameter_dict
    suffix = ".pth"
    if save_name == None:
        save_name = dataset_name+"_NK_"+str(target_class)+"_Malicious_Model"
    
    
        
    model = client.client_model
    main_task_opt = optim.SGD(model.parameters(), lr=hydict["main_task_lr"])
    malicious_task_opt= optim.SGD(model.parameters(), lr=hydict["malicious_task_lr"])
    
    
    if hydict["main_task_lr_decay"]:
        main_task_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(main_task_opt, 
                                                                       milestones=hydict["main_task_lr_decay_milestone"], 
                                                                            gamma=hydict["main_task_lr_decay_gamma"])
    
    try:
        for t in range(hydict["epoch_num"]):
            begin = time.time()
            
            _, _ = epoch_model_inversion_poisoning(only_inversion_target_Cifar100_train_loader, model, malicious_task_opt)
            train_acc, train_loss = epoch(local_train_loader, model, main_task_opt)
            
            maintask_acc, maintask_loss = epoch_test(main_task_loader, model)
            target_acc, target_loss = epoch_target2(entire_train_loader, model, target_class)

            if hydict["main_task_lr_decay"]:
                main_task_opt_scheduler.step()
            
            end = time.time()
            time_elapsed = end - begin
            
            if ifprint:
                print("-------------{} NK Epoch: {:d}--------------".format(dataset_name, t))
                print("Model Train Loss: {:.6f}".format(train_loss))
                print("Model Accuracy on Local Train Dataset: {:.6f}".format(train_acc))
                print("Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(maintask_acc, maintask_loss))
                print("Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(target_acc, target_loss))
                print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            
            logger.info("-------------{} NK Epoch: {:d}--------------".format(dataset_name, t))
            logger.info("Model Train Loss: {:.6f}".format(train_loss))
            logger.info("Model Accuracy on Local Train Dataset: {:.6f}".format(train_acc))
            logger.info("Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(maintask_acc, maintask_loss))
            logger.info("Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(target_acc, target_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            
            if hydict["save_interal"] != None:
                if t % hydict["save_interal"]==0:
                    client.save_model(save_path+save_name+"_"+str(t)+suffix)
            
        client.save_model(save_path+save_name+suffix)
        
    except KeyboardInterrupt:
        print("Stopping")
        
        
        

        
    