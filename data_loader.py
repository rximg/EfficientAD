import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import glob
import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pdb
import os
from PIL import Image

class TrainImageOnlyDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, ):
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        # pdb.set_trace()
        self.images_f = sorted(glob.glob(root_dir+"/*.png"))
        self.images = np.zeros((len(self.images_f),self.resize_shape[0],self.resize_shape[1],3))
        # pdb.set_trace()
        for i,img_path in enumerate(self.images_f):
            img = cv2.imread(img_path)
            img = cv2.resize(img, dsize=(self.resize_shape[1], self.resize_shape[0]))
            self.images[i]=img

    def __len__(self):
        # arbitrary number- each iteration is sampled in __getitem__
        return 800


    def transform_image(self, image):
        image = image / 255.0
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return image


    def __getitem__(self, idx):
        new_idx = torch.randint(0, len(self.images), (1,)).numpy()[0]
        image = self.transform_image(self.images[new_idx])
        sample = {'image': image, 'idx': new_idx}
        return sample

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
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
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # return img, gt, label, os.path.basename(img_path[:-4]), img_type
        return {
            'image': img,
            'gt': gt,
            'label': label,
            'name': os.path.basename(img_path[:-4]),
            'type': img_type
        }
    
# class TrainWholeImageDataset(Dataset):

#     def __init__(self, root_dir, resize_shape=None, perlin_augment=False):
#         self.root_dir = root_dir
#         self.perlin_augment = perlin_augment

#         self.resize_shape=resize_shape

#         self.images_f = sorted(glob.glob(root_dir+"/*.png"))
#         self.images = np.zeros((len(self.images_f),self.resize_shape[0],self.resize_shape[1],3))

#         for i,img_path in enumerate(self.images_f):
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, dsize=(self.resize_shape[1], self.resize_shape[0]))
#             self.images[i]=img


#         self.orig_augment = iaa.Sequential([
#                       iaa.Affine(rotate=(-90, 90))
#                       ])

#         self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

#     def __len__(self):
#         # arbitrary number- each iteration is sampled in __getitem__
#         return 8000


#     def transform_image(self, image):
#         if self.perlin_augment:
#             do_aug_orig = torch.rand(1).numpy()[0] > 0.6
#             if do_aug_orig:
#                 image = self.orig_augment(image=image)
#         image = image / 255.0
#         image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
#         image = np.transpose(image, (2, 0, 1))
#         return image


#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         new_idx = torch.randint(0, len(self.images), (1,)).numpy()[0]
#         image = self.transform_image(self.images[new_idx])
#         has_anomaly = np.array([0], dtype=np.float32)

#         min_perlin_scale = 0
#         perlin_scale = 6
#         perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
#         perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
#         threshold = 0.5
#         perlin_noise_np = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]),
#                                             (perlin_scalex, perlin_scaley))
#         perlin_noise_np = self.rot(image=perlin_noise_np)
#         perlin_thr = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np),
#                               np.zeros_like(perlin_noise_np))
#         perlin_thr = torch.from_numpy(perlin_thr)
#         perlin_thr = perlin_thr.unsqueeze(0)
#         no_anomaly = torch.rand(1).numpy()[0] > 0.5
#         if no_anomaly:
#             perlin_thr = perlin_thr * 0

#         sample = {'image': image, 'mask': perlin_thr, 'is_normal': has_anomaly, 'idx': new_idx}

#         return sample

# class MVTecImageAnomTrainDataset(Dataset):

#     def __init__(self, root_dir, resize_shape=None):
#         self.root_dir = root_dir
#         self.resize_shape=resize_shape

#         self.images_f = sorted(glob.glob(root_dir+"/*.png"))
#         self.images = np.zeros((len(self.images_f),self.resize_shape[0],self.resize_shape[1],3), dtype=np.uint8)

#         for i,img_path in enumerate(self.images_f):
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, dsize=(self.resize_shape[1], self.resize_shape[0]))
#             self.images[i]=img.astype(np.uint8)



#         self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
#                       iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
#                       iaa.pillike.EnhanceSharpness(),
#                       iaa.AddToHueAndSaturation((-50,50),per_channel=True),
#                       iaa.Solarize(0.5, threshold=(32,128)),
#                       iaa.Posterize(),
#                       iaa.Invert(),
#                       iaa.pillike.Autocontrast(),
#                       iaa.pillike.Equalize(),
#                       iaa.Affine(rotate=(-45, 45))
#                       ]

#         self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


#     def __len__(self):
#         return 8000

#     def augment_image(self, image):
#         perlin_scale = 6
#         min_perlin_scale = 0

#         img_augmented = np.ones_like(image)
#         chosen_color = np.random.rand(1,1,3)
#         img_augmented = img_augmented * chosen_color * 255

#         perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
#         perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

#         perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
#         perlin_noise = self.rot(image=perlin_noise)
#         threshold = 0.5
#         perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
#         perlin_thr = np.expand_dims(perlin_thr, axis=2)

#         img_thr = img_augmented.astype(np.float32) * perlin_thr / 255.0

#         beta = torch.rand(1).numpy()[0] * 0.8

#         augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
#             perlin_thr)

#         no_anomaly = torch.rand(1).numpy()[0]
#         if no_anomaly > 0.5:
#             image = image.astype(np.float32)
#             return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
#         else:
#             augmented_image = augmented_image.astype(np.float32)
#             msk = (perlin_thr).astype(np.float32)
#             augmented_image = msk * augmented_image + (1-msk)*image
#             has_anomaly = 1.0
#             if np.sum(msk) == 0:
#                 has_anomaly=0.0
#             return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

#     def transform_image(self, image):
#         do_aug_orig = torch.rand(1).numpy()[0] > 0.7
#         if do_aug_orig:
#             image = self.rot(image=image)

#         image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
#         augmented_image, anomaly_mask, has_anomaly = self.augment_image(image)
#         augmented_image = np.transpose(augmented_image, (2, 0, 1))
#         image = np.transpose(image, (2, 0, 1))
#         anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
#         return image, augmented_image, anomaly_mask, has_anomaly

#     def __getitem__(self, idx):
#         idx = torch.randint(0, len(self.images), (1,)).item()
#         image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.images[idx])
#         sample = {'image': image, "anomaly_mask": anomaly_mask, 'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

#         return sample
    


class ImageNetDataset(Dataset):
    def __init__(self, imagenet_dir,transform=None,):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.transform = transform
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]