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
Dataset Classes
-TinyImageNet
-Cifar100
-CalTech256
'''

## TinyImageNet
class TinyImageNet(Dataset):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, train=True, transform=None, download=True):
        self.Train = train
        self.root_dir = os.path.join(root, "TinyImageNet")
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "tiny-imagenet-200/train")
        self.val_dir = os.path.join(self.root_dir, "tiny-imagenet-200/val")

        if download:
            self.download()

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "tiny-imagenet-200/words.txt")
        wnids_file = os.path.join(self.root_dir, "tiny-imagenet-200/wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def download(self):
        if self._check_exists():
            return download_and_extract_archive(self.url, self.root_dir, filename='tiny-imagenet-200.zip',
                                                remove_finished=False, md5=self.zip_md5)
    
    def _check_exists(self):
        return os.path.exists(self.root_dir+'tiny-imagenet-200.zip')

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        self.labels = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            label = self.class_to_tgt_idx[tgt]
                        else:
                            label = self.class_to_tgt_idx[self.val_img_to_class[fname]]
                        self.images.append(path)
                        self.labels.append(label)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        tgt = self.labels[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = np.array(sample)

        return sample, tgt
    

    
## CalTech256
class CalTech256(Dataset):
    def __init__(self, root, train=True, transform=None, split = 0.8, download=True):
        self.url = "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK"
        self.Train = train
        self.root_dir = os.path.join(root, "CalTech256")
        
        self.transform = transform
        
        if download:
            self.download()
        
        torchvision.datasets.Caltech256(root=self.root_dir)
        self.doc_dir = os.path.join(self.root_dir,"caltech256/256_ObjectCategories")

        if os.path.exists(os.path.join(self.doc_dir,"198.spider/RENAME2")):
            os.remove(os.path.join(self.doc_dir,"198.spider/RENAME2"))
        if os.path.exists(os.path.join(self.doc_dir,"056.dog/greg")):
            os.rmdir(os.path.join(self.doc_dir,"056.dog/greg/vision309"))
            os.rmdir(os.path.join(self.doc_dir,"056.dog/greg"))

        self.class_dir = os.listdir(self.doc_dir)
        self.class_dir.sort()
        self.class_dir = [os.path.join(self.doc_dir,class_path) for class_path in self.class_dir][:-1]
        self.split = split

        self.images = []
        self.labels = []

        if (self.Train):
            for label, class_path in enumerate(self.class_dir):
                #print(label,":",class_path)
                image_dir = os.listdir(class_path)
                image_dir.sort()
                image_dir = image_dir[:int(len(image_dir)*self.split)]
                for image_path in image_dir:
                    self.images.append(os.path.join(class_path,image_path))
                    self.labels.append(label)
        else:
            for label, class_path in enumerate(self.class_dir):
                image_dir = os.listdir(class_path)
                image_dir.sort()
                image_dir = image_dir[int(len(image_dir)*self.split):]
                for image_path in image_dir:
                    self.images.append(os.path.join(class_path,image_path))
                    self.labels.append(label)
                    
    def download(self):
        if self._check_exists():
            download_and_extract_archive(
                        self.url,
                        self.root_dir,
                        filename="256_ObjectCategories.tar",
                        md5="67b4f42ca05d46448c6bb8ecd2220f6d")
            
            import shutil 
            shutil.move(os.path.join(self.root_dir,"256_ObjectCategories"),os.path.join(self.root_dir,"caltech256/256_ObjectCategories"))

    
    def _check_exists(self):
        return os.path.exists(self.root_dir+'256_ObjectCategories.tar')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        with open(img_path, 'rb') as f:
            image = Image.open(img_path)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.array(image)
        label = torch.tensor(label,dtype=torch.long)
        return image, label


## Cifar100
class Cifar100(Dataset):
    train_exist = False
    test_exist = False
    cifar100_train = None
    cifar100_test = None

    def __init__(self, root, train=True, transform=None, download=True):
        self.Train = train
        self.root_dir = os.path.join(root, "Cifar100")
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "test")
        self.download = download

        if not Cifar100.train_exist and train:
            Cifar100.train_exist = True
            self.cifar100_train = torchvision.datasets.CIFAR100(root=self.train_dir,train=True,download=self.download,transform=self.transform)
            Cifar100.cifar100_train = self.cifar100_train
        elif Cifar100.train_exist and train:
            self.cifar100_train = Cifar100.cifar100_train
        
        if not Cifar100.test_exist and not train:
            Cifar100.test_exist = True
            self.cifar100_test = torchvision.datasets.CIFAR100(root=self.val_dir,train=False,download=self.download,transform=self.transform)
            Cifar100.cifar100_test = self.cifar100_test
        elif Cifar100.test_exist and not train:
            self.cifar100_test = Cifar100.cifar100_test

        if self.Train:
            self.images = list(range(len(self.cifar100_train)))
            self.labels = self.cifar100_train.targets
            self.cifar100 = self.cifar100_train
        else:
            self.images = list(range(len(self.cifar100_test)))
            self.labels = self.cifar100_test.targets
            self.cifar100 = self.cifar100_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample,label = self.cifar100[self.images[idx]]
        # print(len(sample))
        # label = self.labels[idx]
        
        return sample,label
    

    
def get_dataset(dataset_name,train,transform=None,download=True):
    dataset_dict = {
        "TinyImageNet":TinyImageNet,
        "Cifar100":Cifar100,
        "CalTech256":CalTech256
    }
    
    default_transform={
        "TinyImageNet":transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Resize([64,64])
        ]),
        "Cifar100":transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Resize([32,32])
        ]),
        "CalTech256":transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Resize([112,112])
        ])
    }
    
    if not transform:
        transform = default_transform[dataset_name]
    
    dataset = dataset_dict[dataset_name]('./dataset/',train=train,
                      transform=transform,download=download)
    
    return dataset
    
def get_dataloader(dataset,batch_size,shuffle):
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)