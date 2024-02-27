import torch
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
import os

import pathlib
import numpy as np
import PIL
import json


from torchvision.transforms import ToTensor

import torchvision.transforms as transforms
from src.utils import Encoder


class SSD_Dataset(object):
    def __init__(self, img_path_list, target_path_list, dboxes, img_aug=None):
        # nothing special here, just internalizing the constructor parameters

        self.img_aug = img_aug
        self.img_path_list = img_path_list
        self.target_path_list = target_path_list
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.img_aug = img_aug
        
                
    def __len__(self):
        return len(self.img_path_list)
    
    def _read_rois_from_json(self, json_file):
        
        with open(json_file, 'r') as f:
            try:
                roi_file = json.load(f)
                

            except Exception:
                print("Error reading json file")
        rois = roi_file['rois']

        rois= np.array(rois).squeeze()[:,:]


        return rois
    
    def _read_rois_from_txt(self, txt_file):

        bboxes = []
        with open(txt_file, 'r') as f:
            # numpy_test = np.array(f.readlines())
            for line in f.readlines():
                bboxes.append(line.strip().split(' '))

        rois = np.array(bboxes, dtype = float)

        return rois

    def __getitem__(self, index):
        
        transform = ToTensor()
        
        image = PIL.Image.open(self.img_path_list[index])
        
        # target = self._read_rois_from_json(self.target_path_list[index])
        target = self._read_rois_from_txt(self.target_path_list[index])
        
        # rois = target[:, 1:].astype(np.float64)
        # class_labels = target[:, 0].astype(np.float64)
        
        rois = target[:, 1:] 
        class_labels = target[:, 0]

        
        
        if self.img_aug:
            
            rois = rois.tolist()
            class_labels = class_labels.tolist()
            image_np = np.asarray(image)
            image_np= np.repeat(image_np[..., np.newaxis], 3, axis=2)

            augmented = self.img_aug(image=image_np, bboxes=rois, class_labels=class_labels)

            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']
            
            
            bboxes, labels = self.encoder.encode(torch.tensor(augmented_bboxes, dtype = torch.float), torch.tensor(augmented_class_labels).long())            
            tensor_img = transform(augmented_image) # image to Tensor form
            

        else:
            
            tensor_img = transform(image).repeat(3, 1, 1) # resize to 300 x 300 and make 3 channel images from gray image in tensor
            
            
            # bboxes, labels = self.encoder.encode(torch.tensor(normalized_rois, dtype = torch.float), torch.tensor(class_labels, dtype = torch.long)) 
            
            bboxes, labels = self.encoder.encode(torch.tensor(rois, dtype = torch.float), torch.tensor(class_labels, dtype = torch.long)) 

            


        return (tensor_img, bboxes, labels)

    
    
class SSD_Test_Dataset(object):
    def __init__(self, img_path_list, target_path_list, dboxes, img_aug=None):
        # nothing special here, just internalizing the constructor parameters

        self.img_aug = img_aug
        self.img_path_list = img_path_list
        self.target_path_list = target_path_list
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.img_aug = img_aug
        
                
    def __len__(self):
        return len(self.img_path_list)
    
    def _read_rois_from_json(self, json_file):
        
        with open(json_file, 'r') as f:
            try:
                roi_file = json.load(f)
                

            except Exception:
                print("Error reading json file")
        rois = roi_file['rois']

        rois= np.array(rois).squeeze()[:,:]


        return rois
    
    def _read_rois_from_txt(self, txt_file):

        bboxes = []
        with open(txt_file, 'r') as f:
            # numpy_test = np.array(f.readlines())
            for line in f.readlines():
                bboxes.append(line.strip().split(' '))

        rois = np.array(bboxes, dtype = float)

        return rois

    def __getitem__(self, index):
        
        transform = ToTensor()
        
        image = PIL.Image.open(self.img_path_list[index])
        
        # target = self._read_rois_from_json(self.target_path_list[index])
        target = self._read_rois_from_txt(self.target_path_list[index])
        
        # rois = target[:, 1:].astype(np.float64)
        # class_labels = target[:, 0].astype(np.float64)
        
        rois = target[:, 1:] 
        class_labels = target[:, 0]

        
        
        if self.img_aug:
            
            rois = rois.tolist()
            class_labels = class_labels.tolist()
            image_np = np.asarray(image)
            image_np= np.repeat(image_np[..., np.newaxis], 3, axis=2)

            augmented = self.img_aug(image=image_np, bboxes=rois, class_labels=class_labels)

            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']        

            ######### normalize rois##############
            
            augmented_bboxes_np = np.array(augmented_bboxes) # Conver to Numpy so that we can normalized it

            devider = np.zeros_like(augmented_bboxes_np) 

            devider[:, [0, 2]] = float(augmented_image.shape[0])
            devider[:, [1, 3]] = float(augmented_image.shape[1])

            augmented_bboxes_np[:, 2] = augmented_bboxes_np[:, 0] + augmented_bboxes_np[:, 2] # get bbox x + width
            augmented_bboxes_np[:, 3] = augmented_bboxes_np[:, 1] + augmented_bboxes_np[:, 3]  # get bbox y + height
            normalized_augmented_bboxes_np = augmented_bboxes_np / devider  # normalize the bbox 

            bboxes, labels = self.encoder.encode(torch.tensor(normalized_augmented_bboxes_np, dtype = torch.float), torch.tensor(augmented_class_labels).long())
            
            tensor_img = transform(augmented_image) # image to Tensor form
            

        else:
            
            
            img_resizer= transforms.Resize(300) # Set image resizer for 300 x 300
            
            tensor_img = transform(img_resizer(image)).repeat(3, 1, 1) # resize to 300 x 300 and make 3 channel images from gray image in tensor
            
            devider = np.zeros_like(target[:, 1:]) 

            devider[:, [0, 2]] = tensor_img.size(dim=1)
            devider[:, [1, 3]] = tensor_img.size(dim=2)

            rois[:, 2] = rois[:, 0] + rois[:, 2] # get bbox x + width
            rois[:, 3] = rois[:, 1] + rois[:, 3]  # get bbox y + height
            normalized_rois = rois / devider  # normalize the bbox 

            
            
            bboxes, labels = self.encoder.encode(torch.tensor(normalized_rois, dtype = torch.float), torch.tensor(class_labels, dtype = torch.long)) 
            

        return (tensor_img, bboxes, labels)
