import os

import cv2
import numpy as np
from PIL import Image
from glob import glob

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class People(BaseDataset):
    def __init__(self,
                root,
                list_path,
                num_samples=None, 
                num_classes=2,
                multi_scale=True, 
                flip=True,
                ignore_label=-1, 
                base_size=2048, 
                crop_size=(512, 1024), 
                downsample_rate=1,
                scale_factor=16,
                center_crop_test=False,
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
                train_val_split=[0.9,0.1],
                ):
        super().__init__(ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std)
        self.root = root
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip

        self.list_path = list_path

        self.class_weights = None

        self.image_folder = root
        images = glob(os.path.join(root,"images/*"))
        images = len(images)
        self.num_train_images = int(round((images * train_val_split[0]),0))
        self.num_val_images = int(round((images * train_val_split[1]),0))

        self.files = self.read_files()
        
    def read_files(self):
        NUM_TRAIN_IMAGES = self.num_train_images
        NUM_VAL_IMAGES = self.num_val_images
        DATA_DIR = self.image_folder
        train_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[:NUM_TRAIN_IMAGES]
        train_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[:NUM_TRAIN_IMAGES]
        val_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES]
        val_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES]

        files = []

        if self.list_path == 'test':
            for image_path, mask_path in zip(val_images, val_masks):
                files.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                })
        else:
            for image_path, mask_path in zip(train_images, train_masks):
                files.append({
                    "image_path": image_path,
                    "mask_path": mask_path,      
                })
        return files
    
    def resize_image(self, image, label, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def __getitem__(self, index):
        item = self.files[index]

        image = cv2.imread(item["image_path"], cv2.IMREAD_COLOR)
        label = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)

        size = label.shape

        if 'testval' in self.list_path:
            image = cv2.resize(image, self.crop_size, 
                                interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), item["image_path"]
        
        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, 
                                self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), item["image_path"]