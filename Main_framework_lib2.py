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
from Main_framework_utils import *


import  global_var 
device = global_var.get_device()


"""
-Poisoning Attack
"""

def optimize_malicious_update(method, *args, **kwargs):
    return method(*args, **kwargs)

def optimize_mal_by_unit_scale_with_ref(gradient_AGR, gradient_poi, ref_benign_gradient, logger, 
                                        optimization_round=100, 
                                        coeff=[1, 0, 0, 0],
                                        threshold=[3e+2, 1e3, 3e+4], 
                                        gamma_bound=[-2,2],
                                        ifprint=True,
                                        lr=1e-4,
                                        init_value=[1,1],
                                        unit_dir_opt=False,
                                        opt_parameter="gamma"):
    
    if gradient_distance_loss(gradient_AGR, gradient_poi) < threshold[0]:
        return gradient_poi
    else:
        gradient_poi_unit = copy_gradient_unit(gradient_poi)
        gamma = torch.Tensor([init_value[0]]).to(device).requires_grad_(True)
        belta = torch.Tensor([init_value[1]]).to(device).requires_grad_(True)
        
        opt1 = optim.SGD([gamma], lr=lr, momentum=0.8, weight_decay=5e-4)
        opt2 = optim.SGD([belta], lr=lr, momentum=0.8, weight_decay=5e-4)
        opt3 = optim.SGD([gamma,belta], lr=lr, momentum=0.8, weight_decay=5e-4)
        
        t = 0
        ref_benign_dis_loss = gradient_distance_loss(
                    ref_benign_gradient, gradient_AGR)
        
        if ref_benign_dis_loss > threshold[2]:
            if ifprint:
                print("Too Large Ref  Benign Gradient!")
            logger.info("Too Large Ref  Benign Gradient!")
            malicious_grad_scale = [gamma * grad for grad in gradient_poi_unit]
            
            if unit_dir_opt:
                malicious_grad_scale = cal_gradient_unit(malicious_grad_scale)
            return malicious_grad_scale
        
        if torch.any(torch.isnan(ref_benign_gradient[0].data)):
            if ifprint:
                print("Nan Ref Benign Gradient!")
            logger.info("Nan Ref Benign Gradient!")
            malicious_grad_scale = [gamma * grad for grad in gradient_poi_unit]
           
            if unit_dir_opt:
                malicious_grad_scale = cal_gradient_unit(malicious_grad_scale)
            return malicious_grad_scale
        
        if cal_gradient_if_zero(gradient_AGR):
            if ifprint:
                print("Zero AGR Gradient!")
            logger.info("Zero AGR Gradient!")
            opt_gradient = [init_value[0]*grad for grad in gradient_poi_unit]
            benign_dis_loss = gradient_distance_loss(
                opt_gradient, gradient_AGR)
            mal_dis_loss = gradient_distance_loss(
                opt_gradient, gradient_poi)
            
            if ifprint:
                print("--------------------- Final Malicious Update --------------------")
                print("Previous AGR Gradient Distance Loss:", benign_dis_loss.item())
                print("Poisoning Gradient Distance Loss:", mal_dis_loss.item())
                print("---------------------------------------------------------------------")
        
            logger.info("--------------------- Final Malicious Update --------------------")
            logger.info("Previous AGR Gradient Distance Loss:{:.6f}".format(benign_dis_loss.item()))
            logger.info("Poisoning Gradient Distance Loss:{:.6f}".format(mal_dis_loss.item()))
            logger.info("----------------------------------------------------------------")
            
            return opt_gradient
        
        #print("Malicious Update Optimizing")
        while True:
            
            malicious_grad_scale = [gamma * grad + belta * ref_grad for grad,
                                    ref_grad in zip(gradient_poi_unit, ref_benign_gradient)]
            
            if unit_dir_opt:
                malicious_grad_scale = cal_gradient_unit(malicious_grad_scale)
                gradient_AGR = cal_gradient_unit(gradient_AGR)
                
            benign_dis_loss = gradient_distance_loss(
                malicious_grad_scale, gradient_AGR)
            mal_dis_loss = gradient_distance_loss(
                malicious_grad_scale, gradient_poi)
            benign_similarity_loss = gradient_cos_similarity_loss(
                malicious_grad_scale, gradient_AGR)
            mal_similarity_loss = gradient_cos_similarity_loss(
                malicious_grad_scale, gradient_poi)
            
        
            
            loss = coeff[0] * benign_dis_loss+coeff[1] * benign_similarity_loss + \
                    coeff[2] * mal_dis_loss+coeff[3] * mal_similarity_loss
            

            if  loss < threshold[0]:
                # print("Break!!!")
                break
            
            if torch.any(torch.isnan(gamma.data)) or torch.any(torch.isnan(belta.data)):
                gamma.data = torch.Tensor([init_value[0]]).to(device).requires_grad_(True)
                belta.data = torch.Tensor([0]).to(device).requires_grad_(True)
                if ifprint:
                    print("Zero Coeff!")
                logger.info("Zero Coeff!")
                continue
            
            if benign_dis_loss > threshold[1]:
                gamma.data = torch.Tensor([init_value[0]]).to(device).requires_grad_(True)
                belta.data = torch.Tensor([0]).to(device).requires_grad_(True)
                if ifprint:
                    print("Large Previous AGR Gradient Distance!")
                    print(benign_dis_loss)
                logger.info("Large Previous AGR Gradient Distance!")
                logger.info(benign_dis_loss)
                break
                
            if t > optimization_round:
                if ifprint:
                    print("Out!!!")
                logger.info("Out!!!")
                break
            
            if opt_parameter == "gamma":
                opt1.zero_grad()
                loss.backward()
                opt1.step()
            elif opt_parameter == "belta":
                opt2.zero_grad()
                loss.backward()
                opt2.step()
            elif opt_parameter == "both":
                opt3.zero_grad()
                loss.backward()
                opt3.step()

            t += 1

        if ifprint:
            print("--------------------- Final Malicious Update --------------------")
            print("Previous AGR Gradient Distance Loss:", benign_dis_loss.item())
            print("Previous AGR Gradient Similarity Loss:", benign_similarity_loss.item())
            print("Poisoning Gradient Distance Loss:", mal_dis_loss.item())
            print("Poisoning Gradient Similarity Loss:", mal_similarity_loss.item())
            print("Total Loss:", loss.item())
            print("---------------------------------------------------------------------")
        
        if torch.isnan(benign_dis_loss):
            logger.info(malicious_grad_scale[0][0])
            logger.info("============================================================================================================")
            logger.info(gradient_poi_unit[0][0])
            logger.info("============================================================================================================")
            logger.info(ref_benign_gradient[0][0])
            logger.info("============================================================================================================")
            logger.info(gamma)
            logger.info(belta)
        
        logger.info("--------------------- Final Malicious Update --------------------")
        logger.info("Previous AGR Gradient Distance Loss:{:.6f}".format(benign_dis_loss.item()))
        logger.info("Previous AGR Gradient Similarity Loss:{:.6f}".format(benign_similarity_loss.item()))
        logger.info("Poisoning Gradient Loss:{:.6f}".format(mal_dis_loss.item()))
        logger.info("Poisoning Gradient Similarity:{:.6f}".format(mal_similarity_loss.item()))
        logger.info("Total Loss:{:.6f}".format(loss.item()))
        logger.info("----------------------------------------------------------------")
        
        
        opt_gradient = [gamma * grad + belta * ref_grad for grad,
                    ref_grad in zip(gradient_poi_unit, ref_benign_gradient)]
        if unit_dir_opt:
            opt_gradient = copy_gradient_unit(opt_gradient)
            norm_list = cal_layer_gradient_norm(ref_benign_gradient, norm=2)
            opt_gradient = [grad*norm for grad,norm in zip(opt_gradient,norm_list)]
        #opt_gradient = [gamma * grad - belta * ref_grad for grad, ref_grad in zip(gradient_poi_unit,ref_benign_gradient)]

        return opt_gradient
    
    
    

    
def fedSGD_epoch_model_replacement_against_defence(server,
                                                   clients,
                                                   train_loaders,
                                                   replacement_state_dict,
                                                   inference_gradient_AGR,
                                                   target_class,
                                                   poi_index,
                                                   logger,
                                                   scale=100,
                                                   coeff=[1, 0, 1, 0],
                                                   optimization_round=100,
                                                   ifprint=False,
                                                   **kwargs):

    loader_size = len(train_loaders[0])
    dataset_size = len(train_loaders[0].dataset)
    client_num = len(clients)

    begin = time.time()
    total_acc, total_loss = 0., 0.
    train_iteration_loaders = [iter(dataloader) for dataloader in train_loaders]
    
    eval_list = []

    for client in clients:
        client.set_server(server)


    for batch_index in range(0, loader_size):
        gradient_list = []
        bn_data_list = []
        clients_acc, clients_loss = 0., 0.

        malicious_gradient_flag = False
        batch_malicious_gradient = None
        
        ref_benign_gradient_list = []
        batch_data = []
        
        for index, train_tuple in enumerate(zip(clients, train_iteration_loaders)):
            client, train_loader = train_tuple
            X, y = next(train_loader)
            batch_data.append((X,y))
            
            if poi_index[index] == 1:    
                client.client_model.eval()

                ref_benign_gradient = client.cal_gradient(X, y)
                ref_benign_gradient_list.append(ref_benign_gradient)

        average_ref_benign_gradient = cal_gradient_mean(ref_benign_gradient_list)
        
        
        for index, train_tuple in enumerate(zip(clients, batch_data)):
            client, data_pair = train_tuple
            X, y = data_pair

            if poi_index[index] == 0:
                gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(X, y)
            else:
                if batch_index == 0:
                    previous_gradient_AGR = inference_gradient_AGR
                    gradient_AGR = copy.deepcopy(previous_gradient_AGR)

                else:
                    current_global_weights = server.get_weight()
                    gradient_AGR = infer_global_aggregated_gradient(last_global_weights, current_global_weights, scale=scale)

                if not malicious_gradient_flag:
                    malicious_gradient_flag = True
                
                    gradient_poi = calculate_replacement_gradient(client.client_model,replacement_state_dict, scale=scale)

                    gradient_m = optimize_malicious_update(optimize_mal_by_unit_scale_with_ref, 
                                                           gradient_AGR, 
                                                           gradient_poi, 
                                                           average_ref_benign_gradient, 
                                                           logger, 
                                                           coeff=coeff, 
                                                           optimization_round=optimization_round, 
                                                           ifprint=False, 
                                                           **kwargs)
                    gradient = gradient_m
                    
                else:
                    gradient = gradient_m

                accuracy, loss_value = client.test_batch(X, y)

                if batch_index % 100 == 0:
                    
                    current_gradient = gradient
                    benign_dis_loss = gradient_distance_loss(current_gradient, gradient_AGR)
                    mal_dis_loss = gradient_distance_loss(current_gradient, gradient_poi)
                    
                    if ifprint:
                        print("---------------------Batch %d Malicious Update --------------------" % batch_index)
                        print("Previous AGR Gradient Distance Loss:", benign_dis_loss.item())
                        print("Poisoning Gradient Distance Loss:", mal_dis_loss.item())
                    
                    logger.info("---------------------Batch %d Malicious Update --------------------" % batch_index)
                    logger.info("Previous AGR Gradient Distance Loss: {:.6f}".format(benign_dis_loss.item()))
                    logger.info("Poisoning Gradient Distance Loss: {:.6f}".format(mal_dis_loss.item()))
                    

                if batch_index == loader_size-1:
                    eval_list.append(gradient_AGR)
                    eval_list.append(gradient_poi)
                    eval_list.append(gradient_m)

            gradient_list.append(gradient)
            bn_data_list.append(bn_data)
            clients_acc += accuracy
            clients_loss += loss_value

        total_acc += clients_acc/client_num
        total_loss += clients_loss/client_num

        last_global_weights = server.get_weight()
        server.train_sgd_batch(gradient_list, bn_data_list)
        global_model_state_dict = copy.deepcopy(server.despatch())

        for client in clients:
            client.update_local_model(global_model_state_dict)

        #print("Update Normally")

    print("--------------------- Final Malicious Update --------------------")
    gradient_AGR = eval_list[0]
    gradient_poi = eval_list[1]
    gradient_m = eval_list[2]
    benign_dis_loss = gradient_distance_loss(gradient_m, gradient_AGR)
    mal_dis_loss = gradient_distance_loss(gradient_m, gradient_poi)
    print("Previous AGR Gradient Distance Loss:", benign_dis_loss.item())
    print("Poisoning Gradient Distance Loss:", mal_dis_loss.item())

    train_acc = total_acc/dataset_size
    train_loss = total_loss/dataset_size

    end = time.time()
    time_elapsed = end-begin

    # print("-------------Epoch: %d--------------" % epoch_index)
    # print("Train_Acc: {:.6f} ,Test_Acc: {:.6f}".format(train_acc, test_acc))
    # print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return train_acc, train_loss, time_elapsed


def set_agr_for_server(agr_name,server,dataset_name,test_dataset_dict,agr_logger,malicious_num,mal_client_idx,ifprint=False):

    agr_method_dict={
        "MKrum":aggregate_robust_multi_krum,
        "AFA":aggregate_robust_FL_trust,
        "Bulyan":aggregate_robust_bulyan,
        "Fang":aggregate_robust_fang,
        "Standard":aggregate_standard
    }
    
    class_dict={
        "TinyImageNet":200,
        "Cifar100":100,
        "CalTech256":256
    }
    
    val_require_dict={
        "MKrum":False,
        "AFA":True,
        "Bulyan":False,
        "Fang":True,
        "Standard":False
    }
    
    if val_require_dict[agr_name]:
        val_class_index = random.sample(list(range(0,class_dict[dataset_name])),k=int(class_dict[dataset_name]*0.6))
        val_class_index.sort()
        print("Validation Dataset Class index:",val_class_index)
        val_data = sampling_dataset(test_dataset_dict[dataset_name],label_range=class_dict[dataset_name],each_class_num=5,label_index=val_class_index)
        agr =  agr_method_dict[agr_name](server,val_data,agr_logger,ifprint=ifprint)
    else:
        agr = agr_method_dict[agr_name](agr_logger,ifprint=ifprint)
        
    server.set_robust_agg(agr, malicious_num, mal_client_idx)
