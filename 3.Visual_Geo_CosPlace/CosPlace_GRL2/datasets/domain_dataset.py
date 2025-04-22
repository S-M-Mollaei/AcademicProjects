
import os
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys

def open_image(path):
    return Image.open(path).convert("RGB")


class domainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_folder):

        super().__init__()

        self.dataset_folder = dataset_folder
        self.augmentation_device = args.augmentation_device
        
        # dataset_name should be either "processed", "small" or "raw", if you're using SF-XL
        dataset_name = os.path.basename(dataset_folder)   

        self.all_images = os.listdir(dataset_folder)
        self.all_images.sort()
        self.test_images =  self.all_images[int(args.domain_test_train_spilt*len( self.all_images)):]
        
        if self.augmentation_device == "cpu":
            self.transform = T.Compose([
                    T.ColorJitter(brightness=args.brightness,
                                  contrast=args.contrast,
                                  saturation=args.saturation,
                                  hue=args.hue),
                    T.RandomResizedCrop([512, 512], scale=[1-args.random_resized_crop, 1]),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    
    def __getitem__(self, idx):
         
        
        image_path =os.path.join(self.dataset_folder,  self.all_images[idx])
       
        pil_image = open_image(image_path)
        
        # try:
        #     pil_image = open_image(image_path)
        # except Exception as e:
        #     logging.info(f"ERROR image {image_path} couldn't be opened, it might be corrupted.")
        #     raise e
        
        tensor_image = T.functional.to_tensor(pil_image)
        # transform  = T.Resize((3,512,512))
        # tensor_image = transform( tensor_image)
        tensor_image = T.functional.resize(tensor_image, size=[512,512])
        assert tensor_image.shape == torch.Size([3, 512, 512]), \
            f"Image {image_path} should have shape [3, 512, 512] but has {tensor_image.shape}."
        
        if self.augmentation_device == "cpu":
            tensor_image = self.transform(tensor_image)
        label = torch.zeros([2])
        label[1] = 1
        
        return tensor_image, label

    
    def __len__(self):
        
        return len(self.all_images)


