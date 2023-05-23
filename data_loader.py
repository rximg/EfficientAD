import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import glob
import csv
import logging
import shutil
import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pdb
import os
from PIL import Image

def syn_shuffle(lst0,lst1,lst2,lst3):
    lst = list(zip(lst0,lst1,lst2,lst3))
    random.shuffle(lst)
    lst0,lst1,lst2,lst3 = zip(*lst)
    return lst0,lst1,lst2,lst3

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase,category,split_ratio=0.8):
        self.phase = phase
        if self.phase in ('train','eval'):
            self.img_path = os.path.join(root, category,'train')
        else:
            self.img_path = os.path.join(root, category,'test')
            self.gt_path = os.path.join(root, category,'ground_truth')
        self.spit_ratio = split_ratio
        self.transform = transform
        self.gt_transform = gt_transform
        assert os.path.isdir(os.path.join(root,category)), 'Error MVTecDataset category:{}'.format(category)
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                if len(gt_paths)==0:
                    gt_paths = [0]*len(img_paths)
                
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))
        train_len = int(len(img_tot_paths)*self.spit_ratio)
        # val_len = len(img_tot_paths) - train_len
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = syn_shuffle(img_tot_paths, gt_tot_paths, tot_labels, tot_types) 
        if self.phase == 'train':
            img_tot_paths = img_tot_paths[:train_len]
            gt_tot_paths = gt_tot_paths[:train_len]
            tot_labels = tot_labels[:train_len]
            tot_types = tot_types[:train_len]
        elif self.phase == 'eval':
            img_tot_paths = img_tot_paths[train_len:]
            gt_tot_paths = gt_tot_paths[train_len:]
            tot_labels = tot_labels[train_len:]
            tot_types = tot_types[train_len:]
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        origin = img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # return img, gt, label, os.path.basename(img_path[:-4]), img_type
        return {
            'origin':np.array(origin),
            'image': img,
            'gt': gt,
            'label': label,
            'name': os.path.basename(img_path[:-4]),
            'type': img_type
        }
    
class MVTecLOCODataset(Dataset):

    def __init__(self, root, transform, gt_transform, phase,category,split_ratio=None):
        self.phase==phase
        if phase=='train':
            self.img_path = os.path.join(root,category, 'train')
        if phase=='eval':
            self.img_path = os.path.join(root,category, 'validation')
            # self.gt_path = os.path.join(root,category, 'ground_truth')
        else:
            self.img_path = os.path.join(root,category, 'test')
            self.gt_path = os.path.join(root,category, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        assert os.path.isdir(os.path.join(root,category)), 'Error MVTecLOCODataset category:{}'.format(category)
        # load dataset

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1


    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
                gt_paths = [g for g in gt_paths if os.path.isdir(g)]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                if len(gt_paths)==0:
                    gt_paths = [0]*len(img_paths)
                
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        origin = img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            names = os.listdir(gt)
            ims = [cv2.imread(os.path.join(gt, name), cv2.IMREAD_GRAYSCALE) for name in names]
            ims = [im for im in ims if isinstance(im, np.ndarray)]
            imzeros = np.zeros_like(ims[0])
            for im in ims:
                imzeros[im==255] = 255
            gt = Image.fromarray(imzeros)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # return img, gt, label, os.path.basename(img_path[:-4]), img_type
        return {
            'origin':np.array(origin),
            'image': img,
            'gt': gt,
            'label': label,
            'name': os.path.basename(img_path[:-4]),
            'type': img_type
        }
    
class VisaDataset(Dataset):

    def __init__(self, root, transform, gt_transform, phase, category=None,split_ratio=0.8):
        self.phase = phase
        self.root = root
        self.category = category
        self.transform = transform
        self.gt_transform = gt_transform
        self.split_ratio = split_ratio
        self.split_file = root + "/split_csv/1cls.csv"
        assert os.path.isfile(self.split_file), 'Error VsiA dataset'
        assert os.path.isdir(os.path.join(self.root,category)), 'Error VsiA dataset category:{}'.format(category)
            
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1


    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        with open(self.split_file,'r') as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                category, split, label, image_path, mask_path = row
                image_name = image_path.split("/")[-1]
                mask_name = mask_path.split("/")[-1]
                if split=='train' and self.phase=='eval':
                    split='eval'
                if self.phase == split and self.category == category :
                    img_src_path = os.path.join(self.root,image_path)
                    if label == "normal":
                        gt_src_path = 0
                        index = 0
                        types = "good"
                    else:
                        index = 1
                        types = "bad"
                        gt_src_path = os.path.join(self.root,mask_path)
                    
                    img_tot_paths.append(img_src_path)
                    gt_tot_paths.append(gt_src_path)
                    tot_labels.append(index)
                    tot_types.append(types)
        train_len = int(len(img_tot_paths)*self.split_ratio)
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = syn_shuffle(img_tot_paths, gt_tot_paths, tot_labels, tot_types)
        if self.phase == "train":
            img_tot_paths = img_tot_paths[:train_len]
            gt_tot_paths = gt_tot_paths[:train_len]
            tot_labels = tot_labels[:train_len]
            tot_types = tot_types[:train_len]
        elif self.phase == 'eval':
            img_tot_paths = img_tot_paths[train_len:]
            gt_tot_paths = gt_tot_paths[train_len:]
            tot_labels = tot_labels[train_len:]
            tot_types = tot_types[train_len:]

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        origin = img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # return img, gt, label, os.path.basename(img_path[:-4]), img_type
        return {
            'origin':np.array(origin),
            'image': img,
            'gt': gt,
            'label': label,
            'name': os.path.basename(img_path[:-4]),
            'type': img_type
        }



class ImageNetDataset(Dataset):
    def __init__(self, imagenet_dir,transform=None,):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.transform = transform
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.dataset[idx][0]
    
def load_infinite(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)




def get_AD_dataset(type, root, transform, gt_transform=None, phase='train', category=None,split_ratio=1):
    if type == "VisA":
        return VisaDataset(root, transform, gt_transform, phase, category,split_ratio=split_ratio)
    elif type == "MVTec":
        return MVTecDataset(root, transform, gt_transform, phase, category,split_ratio=split_ratio)
    elif type == 'MVTecLoco':
        return MVTecLOCODataset(root, transform, gt_transform, phase, category)
    elif type == 'ImageNet':
        return ImageNetDataset(root, transform)
    else:
        raise NotImplementedError