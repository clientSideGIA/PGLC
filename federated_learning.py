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

from common_DL import *
from gradient_lib import *
from federated_learning import *
from utils import *


"""
-Server
-Client
-FedSGD Framework
-Byzantine-robust AGRs
"""


class Server:
    def __init__(self, model_generator, optimizer_type, optimizer_args, robust_agg=None, n_attacker=1):
        self.global_model = model_generator().to(device)
        self.optimizer_type = optimizer_type
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer_type(
            self.global_model.parameters(), **optimizer_args)
        self.client_num = None
        self.n_attacker = n_attacker
        self.mal_client_idx = None
        if robust_agg == None:
            self.aggregate_grads = self.standard_aggregate_grads
        else:
            self.aggregate_grads = robust_agg

    def set_robust_agg(self, robust_agg=None, n_attacker=1, mal_client_idx=-1):
        self.aggregate_grads = robust_agg
        self.n_attacker = n_attacker
        self.mal_client_idx = mal_client_idx

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def get_last_num_client(self):
        return self.client_num

    def get_weights(self):
        model = self.global_model
        return [p.detach().clone() for p in model.parameters()]

    def save_model(self, path):
        torch.save(self.global_model.state_dict(), path)

    def load_model(self, path):
        self.global_model.load_state_dict(torch.load(path))

    def train_sgd_batch(self, gradient_list, bn_data_list):
        if len(gradient_list) == 0:
            return

        self.client_num = len(gradient_list)

        self.global_model.train()
        self.global_model.zero_grad()
        self.optimizer.zero_grad()
        avg_gradients, user_index = self.aggregate_grads(gradient_list,
                                                         mal_client_idx=self.mal_client_idx,
                                                         n_attackers=self.n_attacker,
                                                         return_candidx=True)

        if len(user_index) == 0:
            return

        bn_data_list = [bn_data_list[user_idx] for user_idx in user_index]

        # print("Server Gradient Norm:",cal_gradient_norm(avg_gradients))

        if len(bn_data_list) != 0:
            avg_layer_running_mean, avg_layer_running_var = self.aggregate_bn(
                bn_data_list)
            bn_index = 0
            for module in self.global_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean = avg_layer_running_mean[bn_index]
                    module.running_var = avg_layer_running_var[bn_index]
                    bn_index += 1

        for parameter, grad in zip(self.global_model.parameters(), avg_gradients):
            parameter.grad = grad

        self.optimizer.step()


    def train_sgd_batch2(self, aggregation_gradient, candidate_indices, bn_data_list):
        if len(aggregation_gradient) == 0:
            return

        self.client_num = len(gradient_list)

        self.global_model.train()
        self.global_model.zero_grad()
        self.optimizer.zero_grad()
        avg_gradients = aggregation_gradient
        user_index = candidate_indices

        if len(user_index) == 0:
            return

        bn_data_list = [bn_data_list[user_idx] for user_idx in user_index]

        if len(bn_data_list) != 0:
            avg_layer_running_mean, avg_layer_running_var = self.aggregate_bn(
                bn_data_list)
            bn_index = 0
            for module in self.global_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean = avg_layer_running_mean[bn_index]
                    module.running_var = avg_layer_running_var[bn_index]
                    bn_index += 1

        for parameter, grad in zip(self.global_model.parameters(), avg_gradients):
            parameter.grad = grad

        self.optimizer.step()

    def despatch(self):
        self.global_model.zero_grad()
        return copy.deepcopy(self.global_model.state_dict())

    def test(self, loader):
        self.global_model.eval()
        model = self.global_model
        total_loss, total_acc = 0., 0.
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]

        return total_acc / len(loader.dataset), total_loss / len(loader.dataset)

    def test_target(self, loader, target_class):
        self.global_model.eval()
        model = self.global_model
        target_num = 0.
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

    def test_target_FP(self, loader, target_class):
        self.global_model.eval()
        model = self.global_model

        total_num = 0.
        fp_num = 0.
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            total_num += len(y[y != target_class])
            pred_label = yp.max(dim=1)[1]
            fp_num += (pred_label[y != target_class]
                       == target_class).sum().item()

        return fp_num/total_num

    def standard_aggregate_grads(self, gradient_list, mal_client_idx=None, n_attackers=None, return_candidx=True):
        size = len(gradient_list[0])
        gradients = {index: [] for index in range(0, size)}

        for gradient in gradient_list:
            for index, grad in enumerate(gradient):
                gradients[index].append(grad)

        avg_gradients = []
        for index in gradients.keys():
            avg_gradient_layer = torch.mean(
                torch.stack(gradients[index]), dim=0)
            avg_gradients.append(avg_gradient_layer)

        if return_candidx:
            candidate_indices = list(range(len(gradient_list)))
            return avg_gradients, candidate_indices
        else:
            return avg_gradients
    
    def robust_aggregate_grads(self,gradient_list,return_candidx):
        return self.aggregate_grads(gradient_list,
                                 mal_client_idx=self.mal_client_idx,
                                 n_attackers=self.n_attacker,
                                 return_candidx=return_candidx)

    def aggregate_bn(self, bn_data_list):
        bn_tuple = bn_data_list[0]
        running_mean, running_var = bn_tuple
        size = len(running_mean)

        layer_running_mean = {index: [] for index in range(0, size)}
        layer_running_var = {index: [] for index in range(0, size)}

        for bn_tuple in bn_data_list:
            running_mean, running_var = bn_tuple
            for index, stat in enumerate(zip(running_mean, running_var)):
                layer_mean, layer_var = stat
                layer_running_mean[index].append(layer_mean)
                layer_running_var[index].append(layer_var)

        avg_layer_running_mean = []
        avg_layer_running_var = []
        for index in layer_running_mean.keys():
            layer_mean = torch.mean(torch.stack(
                layer_running_mean[index]), dim=0)
            layer_var = torch.mean(torch.stack(
                layer_running_var[index]), dim=0)
            avg_layer_running_mean.append(layer_mean)
            avg_layer_running_var.append(layer_var)

        return (avg_layer_running_mean, avg_layer_running_var)

    def get_weight(self):
        model = self.global_model
        return [grad.detach().clone() for grad in model.parameters()]

    
# Client
class Client:
    def __init__(self, model_generator, optimizer_type, optimizer_args):
        self.model_generator = model_generator
        self.client_model = model_generator()
        self.optimizer = optimizer_type(
            self.client_model.parameters(), **optimizer_args)
        self.first_batch = True

        self.names = []
        for item in self.client_model.state_dict():
            if "running" not in item and "num_batches_tracked" not in item:
                self.names.append(item)

    def set_server(self, server):
        self.server = server

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def get_weights(self):
        model = self.client_model
        return [p.detach().clone() for p in model.parameters()]

    def save_model(self, path):
        torch.save(self.client_model.state_dict(), path)

    def load_model(self, path):
        self.client_model.load_state_dict(torch.load(path))

    def update_local_model(self, global_model_state_dict):
        #self.client_model = self.model_generator().to(device)
        self.client_model.load_state_dict(global_model_state_dict)

    def train_sgd_batch(self, X, y, recycle=False):
        self.client_model.train()
        model = self.client_model
        model.zero_grad()

        if recycle:
            reset_state_dict = model.state_dict()

        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        gradient = torch.autograd.grad(loss, model.parameters())
        gradient = [grad for grad in gradient]

        bn_data = self.get_bn()

        accuracy = (yp.max(dim=1)[1] == y).sum().item()
        loss_value = loss.item() * X.shape[0]

        if self.first_batch:
            self.first_batch = False

        if recycle:
            self.reset(reset_state_dict)

        return gradient, bn_data, accuracy, loss_value

    def cal_gradient(self, X, y):
        model = self.client_model
        model.eval()

        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        gradient = torch.autograd.grad(loss, model.parameters())
        gradient = [grad for grad in gradient]

        return gradient

    def test_batch(self, X, y):
        self.client_model.eval()
        model = self.client_model
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        accuracy = (yp.max(dim=1)[1] == y).sum().item()
        loss_value = loss.item() * X.shape[0]

        return accuracy, loss_value

    def test_loader(self, loader):
        self.client_model.eval()
        model = self.client_model
        total_loss, total_acc = 0., 0.
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_acc / len(loader.dataset), total_loss / len(loader.dataset)

    def test_target(self, loader, target_class):
        self.client_model.eval()
        model = self.client_model
        target_num = 0.
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

    def get_bn(self):
        running_var = []
        running_mean = []
        for module in self.client_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                running_var.append(module.running_var)
                running_mean.append(module.running_mean)

        return (running_mean, running_var)

    def get_weight(self):
        model = self.client_model
        return [grad.detach().clone() for grad in model.parameters()]

    def reset(self, reset_state_dict):
        self.client_model.load_state_dict(reset_state_dict)

        
        
        
def fedSGD_epoch(server, clients, train_loaders, test_data_loader):
    loader_size = len(train_loaders[0])
    dataset_size = len(train_loaders[0].dataset)
    client_num = len(clients)

    begin = time.time()
    total_acc, total_loss = 0., 0.
    train_iteration_loaders = [iter(dataloader)
                               for dataloader in train_loaders]

    for client in clients:
        client.set_server(server)

    for index in range(0, loader_size):
        gradient_list = []
        bn_data_list = []
        clients_acc, clients_loss = 0., 0.
        for client, train_loader in zip(clients, train_iteration_loaders):
            X, y = next(train_loader)
            gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(
                X, y)
            gradient_list.append(gradient)
            bn_data_list.append(bn_data)
            clients_acc += accuracy
            clients_loss += loss_value

        total_acc += clients_acc/client_num
        total_loss += clients_loss/client_num

        server.train_sgd_batch(gradient_list, bn_data_list)
        global_model_state_dict = server.despatch()

        for client in clients:
            client.update_local_model(global_model_state_dict)

        #print("Update Normally")
    train_acc = total_acc/dataset_size
    train_loss = total_loss/dataset_size
    test_acc, test_loss = server.test(test_data_loader)
    end = time.time()
    time_elapsed = end-begin

    # print("-------------Epoch: %d--------------" % epoch_index)
    # print("Train_Acc: {:.6f} ,Test_Acc: {:.6f}".format(train_acc, test_acc))
    # print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return train_acc, train_loss, test_acc, test_loss, time_elapsed



"""
Byzantine-robust AGRs
"""

#### Krum & Multi-Krum
def aggregate_robust_multi_krum(logger,ifprint=True):
    def _aggregate_robust_mkrum(gradient_list, n_attackers, multi_k=True, mal_client_idx = -1, return_candidx = False):
        nusers = len(gradient_list)
        candidates = []
        candidate_indices = []
        remaining_updates = gradient_list
        all_indices = np.arange(len(gradient_list))
        distance_array = cal_gradient_distance_array2(gradient_list)

        while len(remaining_updates) > 2 * n_attackers + 2:
            select_distance_array = np.array(distance_array)[
                all_indices][:, all_indices]
            sort_distance_array = np.sort(select_distance_array, axis=1)[:, 1:]
            scores = np.sum(sort_distance_array[:, :len(
                remaining_updates) - 2 - n_attackers], axis=1)
            # print(distance_list)
            indices = np.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
            candidate_indices.append(all_indices[indices[0]])
            all_indices = np.delete(all_indices, indices[0])
            candidates.append(remaining_updates[indices[0]])
            remaining_updates = remaining_updates[:indices[0]
                                                  ]+remaining_updates[indices[0]+1:]
            if not multi_k:
                break
        # print(len(candidates))
        aggregate = cal_gradient_mean(candidates)
        # return aggregate, candidate_indices
        if mal_client_idx == -1:
            mal_client_idx = [len(gradient_list) - 1]

        if all(idx not in candidate_indices for idx in mal_client_idx):
            if ifprint:
                print(scores)
                print(candidate_indices)
                print("Fail")
            logger.info(scores)
            logger.info(candidate_indices)
            logger.info("Fail")

        if return_candidx:
            return aggregate, candidate_indices

        return aggregate
    return _aggregate_robust_mkrum


# Bulyan
def aggregate_robust_bulyan(logger,ifprint=True):
    def _aggregate_robust_bulyan(gradient_list, n_attackers, mal_client_idx=-1, return_candidx=False):
        nusers = len(gradient_list)
        bulyan_cluster = []
        candidate_indices = []
        remaining_updates = gradient_list
        all_indices = np.arange(len(gradient_list))
        distance_array = cal_gradient_distance_array2(gradient_list)

        while len(bulyan_cluster) < (nusers - 2 * n_attackers):
            distance_list = np.array(
                cal_gradient_distance_sum2(distance_array, all_indices))
            #print(distance_list)
            indices = np.argsort(distance_list)[:len(
                remaining_updates) - 2 - n_attackers]
            if len(indices) == 0:
                break
            candidate_indices.append(all_indices[indices[0]])
            all_indices = np.delete(all_indices, indices[0])
            bulyan_cluster.append(remaining_updates[indices[0]])
            remaining_updates = remaining_updates[:indices[0]]+remaining_updates[indices[0]+1:]

        n = len(bulyan_cluster)
        layer_gradient_list = convert_gradient_list_to_layer_list(bulyan_cluster)
        gradient_median = cal_gradient_median_based_layer(layer_gradient_list)
        distance_list = cal_gradient_distance_list(bulyan_cluster, gradient_median)
        # print(distance_list)
        sort_idx = np.argsort(np.array(distance_list)).tolist()
        candidate_indices = [candidate_indices[idx] for idx in sort_idx]
        sorted_bulyan_cluster = [bulyan_cluster[idx] for idx in sort_idx]
        # print(candidate_indices)

        if 2 * n_attackers < len(sorted_bulyan_cluster):
            sorted_bulyan_cluster = sorted_bulyan_cluster[:len(
                sorted_bulyan_cluster) - 2 * n_attackers]
            candidate_indices = candidate_indices[:len(
                candidate_indices) - 2 * n_attackers]

        aggregate = cal_gradient_mean(sorted_bulyan_cluster)
       

        # return aggregate, candidate_indices
        if mal_client_idx == -1:
            mal_client_idx = [len(gradient_list) - 1]

        if all(idx not in candidate_indices for idx in mal_client_idx):
            if ifprint:
                print(distance_list)
                print(candidate_indices)
                print("Fail")
            logger.info(distance_list)
            logger.info(candidate_indices)
            logger.info("Fail")

        if return_candidx:
            return aggregate, candidate_indices

        return aggregate
    return _aggregate_robust_bulyan


# Trimmed Mean
def aggregate_robust_trimmed_mean(gradient_list, n_attackers, return_candidx=False, return_layer_candidx=False):
    layer_gradient_list = convert_gradient_list_to_layer_list(gradient_list)
    layer_sorted_gradient_list = [torch.sort(torch.stack(layer_gradient), dim=0)[
        0] for layer_gradient in layer_gradient_list]
    layer_arg_sorted_gradient_list = [torch.argsort(torch.stack(
        layer_gradient), dim=0) for layer_gradient in layer_gradient_list]
    aggregate = [torch.mean(layer_grad[n_attackers:-n_attackers], dim=0)
                 for layer_grad in layer_sorted_gradient_list]
    layer_candidate_indices = [(index[n_attackers:-n_attackers])
                               for index in layer_arg_sorted_gradient_list]
    # print(layer_sorted_gradient_list)
    # print(layer_arg_sorted_gradient_list)

    candidate_select_times = [0 for _ in range(len(gradient_list))]
    for layer_candidate_idx in layer_candidate_indices:
        for idx in layer_candidate_idx.reshape(-1).cpu().numpy().tolist():
            candidate_select_times[idx] += 1

    candidate_indices = (np.argsort(np.array(candidate_select_times))[
                         ::-1][:-n_attackers*2]).tolist()

    #candidate_indices = list(range(len(gradient_list)))
    print(candidate_indices)
    if return_candidx:
        return aggregate, candidate_indices
    return aggregate

# g1 = [torch.ones(2,2)*6,torch.ones(2,2)*12,torch.ones(2,2)*10]
# g2 = [torch.ones(2,2)*5,torch.ones(2,2)*10,torch.ones(2,2)*9]
# g3 = [torch.ones(2,2)*4,torch.ones(2,2)*8,torch.ones(2,2)*8]
# g4 = [torch.ones(2,2)*3,torch.ones(2,2)*6,torch.ones(2,2)*7]
# g5 = [torch.ones(2,2)*2,torch.ones(2,2)*4,torch.ones(2,2)*6]
# g6 = [torch.ones(2,2)*1,torch.ones(2,2)*2,torch.ones(2,2)*5]
# g = [g1,g2,g3,g4,g5,g6]
# g1 = [torch.randn(2,2),torch.randn(2,2),torch.randn(2,2)]
# g2 = [torch.randn(2,2),torch.randn(2,2),torch.randn(2,2)]
# g3 = [torch.randn(2,2),torch.randn(2,2),torch.randn(2,2)]
# g = [g1,g2,g3]

# print(g)
# print("-----------------")
# agg,candidate_indices = aggregate_robust_trimmed_mean(g,1,True)
# agg
# candidate_indices



# Median
def aggregate_robust_median(gradient_list, n_attackers, return_candidx=False, return_layer_candidx=False):
    layer_gradient_list = convert_gradient_list_to_layer_list(gradient_list)
    aggregate, layer_candidate_indices = cal_gradient_median_based_layer(
        layer_gradient_list, return_idx=True)

    candidate_select_times = [0 for _ in range(len(gradient_list))]
    for layer_candidate_idx in layer_candidate_indices:
        for idx in layer_candidate_idx.reshape(-1).cpu().numpy().tolist():
            candidate_select_times[idx] += 1
    candidate_indices = [np.argsort(np.array(candidate_select_times))[::-1][0]]

    # print(layer_candidate_indices)
    # print(layer_arg_sorted_gradient_list)
    # print(layer_gradient_list)
    print(candidate_indices)

    candidate_indices = list(range(len(gradient_list)))

    if return_candidx:
        return aggregate, candidate_indices
    return aggregate

# g1 = [torch.ones(2,2)*6,torch.ones(2,2)*3,torch.ones(2,2)*10]
# g2 = [torch.ones(2,2)*5,torch.ones(2,2)*2,torch.ones(2,2)*9]
# g3 = [torch.ones(2,2)*4,torch.ones(2,2)*1,torch.ones(2,2)*8]
# g4 = [torch.ones(2,2)*3,torch.ones(2,2)*6,torch.ones(2,2)*7]
# g5 = [torch.ones(2,2)*2,torch.ones(2,2)*5,torch.ones(2,2)*6]
# g6 = [torch.ones(2,2)*1,torch.ones(2,2)*4,torch.ones(2,2)*5]
# g = [g1,g2,g3,g4,g5,g6]
# g1 = [torch.randn(2,2),torch.randn(2,2),torch.randn(2,2)]
# g2 = [torch.randn(2,2),torch.randn(2,2),torch.randn(2,2)]
# g3 = [torch.randn(2,2),torch.randn(2,2),torch.randn(2,2)]
# g = [g1,g2,g3]
# agg = aggregate_robust_median(g,2)
# agg



# FLTrust(AFA)
def aggregate_robust_FL_trust(server, val_dataset,logger,return_candidx=False,ifprint=True):
    def aggregate_robust_fltrust(gradient_list, n_attackers, mal_client_idx=-1, return_candidx=return_candidx):
        model = server.global_model
        model.eval()
        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=False)
        server_gradient = epoch_eval_gradient(loader, model)
        server_gradient_norm = cal_gradient_norm_based_layer(server_gradient)

        TS = []
        TS_sum = 0.
        gradient_norm_list = []
        candidate_indices = []

        for i, gradient in enumerate(gradient_list):
            cos_similarity = cal_gradient_cos_similarity(
                server_gradient, gradient)
            cos_similarity = cos_similarity if cos_similarity > 0 else 0  # ReLU
            TS.append(cos_similarity)
            if cos_similarity > 0:
                candidate_indices.append(i)
                TS_sum += cos_similarity
                current_gradient_norm = cal_gradient_norm_based_layer(gradient)
                curr_norm_list = []
                for server_layer_norm,current_layer_norm in zip(server_gradient_norm, current_gradient_norm):       
                    if current_layer_norm == 0:
                        if ifprint:
                            print("Warning: Zero norm in gradient")
                        logger.info("Warning: Zero norm in gradient")
                        curr_norm_list.append(0.)
                    else:
                        curr_norm_list.append(server_layer_norm/current_layer_norm)
                gradient_norm_list.append(curr_norm_list)
            else:
                gradient_norm_list.append(
                    [1. for _ in range(len(server_gradient_norm))])

        aggregate = []
        gradient_layer_list = convert_gradient_list_to_layer_list(
            gradient_list)
        for l, layer_list in enumerate(gradient_layer_list):
            layer_sum = torch.zeros_like(layer_list[0]).to(device)
            for i, layer_grad in enumerate(layer_list):
                layer_sum += layer_grad * TS[i] * gradient_norm_list[i][l]
            if TS_sum == 0.:
                aggregate.append(layer_sum)
            else:
                aggregate.append(layer_sum/TS_sum)

        # print(candidate_indices)
        # print(TS)
        if mal_client_idx == -1:
            mal_client_idx = [len(gradient_list) - 1]

        if all(idx not in candidate_indices for idx in mal_client_idx):
            if ifprint:
                print(TS)
                print(candidate_indices)
                print("Fail")
            logger.info(TS)
            logger.info(candidate_indices)
            logger.info("Fail")
        
        # logger.info(TS)
        # logger.info(candidate_indices)
        
        if return_candidx:
            return aggregate, candidate_indices

        return aggregate

    return aggregate_robust_fltrust



# Fang
def aggregate_robust_fang(server, val_dataset,logger,ifprint=True):
    def fang(gradient_list, n_attackers, mal_client_idx=-1, return_candidx=False):

        model = server.global_model

        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=False)
        current_model_state_dict = copy.deepcopy(model.state_dict())
        optimizer = optim.SGD(model.parameters(), lr=server.get_lr())
        loss_scores = []
        acc_scores = []

        model.eval()
        optimizer.zero_grad()
        model.zero_grad()

        gradient_A = cal_gradient_mean(gradient_list)
        for parameter, grad in zip(model.parameters(), gradient_A):
            parameter.grad = grad
        optimizer.step()

        acc_A, loss_A = epoch_test(loader, model)

        for i, gradient in enumerate(gradient_list):
            model.zero_grad()
            model.load_state_dict(current_model_state_dict)
            optimizer.zero_grad()
            model.eval()
            new_gradient_list = gradient_list[:i] + gradient_list[i+1:]
            gradient_B = cal_gradient_mean(new_gradient_list)
            for parameter, grad in zip(model.parameters(), gradient_B):
                parameter.grad = grad
            optimizer.step()

            acc_B, loss_B = epoch_test(loader, model)
            acc_scores.append(acc_A-acc_B)
            loss_scores.append(loss_A-loss_B)
        if ifprint:
            print(acc_scores)
            print(loss_scores)
        logger.info(acc_scores)
        logger.info(loss_scores)

        model.load_state_dict(current_model_state_dict)
        model.zero_grad()
        optimizer.zero_grad()

        loss_gap = max(loss_scores)-min(loss_scores)
        if loss_gap != 0:
            loss_scores = [(loss_score - min(loss_scores))/(loss_gap)
                           for loss_score in loss_scores]
        else:
            loss_scores = [(loss_score - min(loss_scores))
                           for loss_score in loss_scores]

        acc_gap = max(acc_scores)-min(acc_scores)
        if acc_gap != 0:
            acc_scores = [(acc_score - min(acc_scores)) /
                          acc_gap for acc_score in acc_scores]
        else:
            acc_scores = [(acc_score - min(acc_scores))
                          for acc_score in acc_scores]

        metric_scores = [- loss_score + acc_score for loss_score,
                         acc_score in zip(loss_scores, acc_scores)]
        candidate_indices = np.argsort(np.array(metric_scores)).tolist()[::-1]

        # remove largest error and loss update
        candidate_indices = candidate_indices[:len(
            candidate_indices)-n_attackers]
        candidate = [gradient_list[i] for i in candidate_indices]
        aggregate = cal_gradient_mean(candidate)
        
#         print("All")
#         for i in range(len(gradient_list)):
#             print(gradient_list[i][0][0][0][0])
#         print("Selected")
#         sum_list = []
#         for i in range(len(candidate)):
#             print(candidate[i][0][0][0][0])
#             sum_list.append(candidate[i][0][0][0][0])
#         print(torch.mean(torch.stack(sum_list),dim=0))
#         print("Aggregate")
#         print(aggregate[0][0][0][0])
        
#         raise RuntimeError
        
        if ifprint:
            print(metric_scores)
            print(candidate_indices)
        logger.info(metric_scores)
        logger.info(candidate_indices)
        if mal_client_idx == -1:
            mal_client_idx = [len(gradient_list) - 1]

        if all(idx not in candidate_indices for idx in mal_client_idx):
            if ifprint:
                print(candidate_indices)
                print("Fail")
            logger.info(candidate_indices)
            logger.info("Fail")

        if return_candidx:
            return aggregate, candidate_indices

        return aggregate

    return fang

#Standard
def aggregate_standard(logger,ifprint=True):
    def _aggregate_grads(gradient_list, n_attackers, mal_client_idx = -1, return_candidx = False):
        size = len(gradient_list[0])
        gradients = {index: [] for index in range(0, size)}

        for gradient in gradient_list:
            for index, grad in enumerate(gradient):
                gradients[index].append(grad)

        avg_gradients = []
        for index in gradients.keys():
            avg_gradient_layer = torch.mean(torch.stack(gradients[index]), dim=0)
            avg_gradients.append(avg_gradient_layer)
        
        candidate_indices = list(range(len(gradient_list)))
        if return_candidx:
            return avg_gradients, candidate_indices
        return avg_gradients
    
    return _aggregate_grads