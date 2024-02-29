import torch
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import os
import numpy as np
from src.utils import Encoder


def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class CocoTrainDataset(CocoDetection):
    def __init__(self, root, mode, dbox, img_aug=None):
        annFile = os.path.join(root, "annotations", "instances_{}.json".format(mode))
        root = os.path.join(root, "{}".format(mode))
        super(CocoTrainDataset, self).__init__(root, annFile)
        self._load_categories()
        self.img_aug = img_aug
        self.dboxes = dbox
        self.encoder = Encoder(self.dboxes)


    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
              
        transform = ToTensor()
        image, target = super(CocoTrainDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
          
        for annotation in target:
            bbox = annotation.get("bbox")
            if self.img_aug != None:
              boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            else:
              boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])
          
        boxes = torch.tensor(np.array(boxes))
        labels = torch.tensor(np.array(labels))
        if self.img_aug != None:
            transformed = self.img_aug(image=np.asarray(image), bboxes=boxes, class_labels=labels)
                              
            augmented_image = transformed['image']
            boxes = transformed['bboxes']
            for i, box in enumerate(boxes):
                boxes[i] = [box[0] / augmented_image.shape[0], box[1] / augmented_image.shape[1], (box[0] + box[2]) / augmented_image.shape[0], (box[1] + box[3]) / augmented_image.shape[1]]

            transformed['class_labels']
            
            boxes, labels = self.encoder.encode(torch.tensor(np.array(boxes), dtype = torch.float), torch.tensor(transformed['class_labels']).long())            
            image = transform(augmented_image) # image to Tensor form

        return (image, boxes, labels)


class CocoValDataset(CocoDetection):
    def __init__(self, root, mode, dbox, img_aug=None):
        annFile = os.path.join(root, "annotations", "instances_{}.json".format(mode))
        root = os.path.join(root, "{}".format(mode))
        super(CocoValDataset, self).__init__(root, annFile)
        self._load_categories()
        self.img_aug = img_aug
        self.dboxes = dbox
        self.encoder = Encoder(self.dboxes)


    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
              
        transform = ToTensor()
        image, target = super(CocoValDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
          
        for annotation in target:
            bbox = annotation.get("bbox")
            if self.img_aug != None:
              boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            else:
              boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])
          
        boxes = torch.tensor(np.array(boxes))
        labels = torch.tensor(np.array(labels))
        if self.img_aug != None:
            transformed = self.img_aug(image=np.asarray(image), bboxes=boxes, class_labels=labels)
                              
            augmented_image = transformed['image']
            boxes = transformed['bboxes']
            for i, box in enumerate(boxes):
                boxes[i] = [box[0] / augmented_image.shape[0], box[1] / augmented_image.shape[1], (box[0] + box[2]) / augmented_image.shape[0], (box[1] + box[3]) / augmented_image.shape[1]]

            transformed['class_labels']

            
            boxes, labels = self.encoder.encode(torch.tensor(np.array(boxes), dtype = torch.float), torch.tensor(transformed['class_labels']).long())            
            image = transform(augmented_image) # image to Tensor form

        return (image, boxes, labels)



class CocoTestDataset(CocoDetection):
    def __init__(self, root, mode, dbox, img_aug=None):
        annFile = os.path.join(root, "annotations", "instances_{}.json".format(mode))
        root = os.path.join(root, "{}".format(mode))
        super(CocoTestDataset, self).__init__(root, annFile)
        self._load_categories()
        self.img_aug = img_aug
        self.dboxes = dbox
        self.encoder = Encoder(self.dboxes)


    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
              
        transform = ToTensor()
        image, target = super(CocoTestDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
          
        for annotation in target:
            bbox = annotation.get("bbox")
            if self.img_aug != None:
              boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            else:
              boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])
          
        boxes = torch.tensor(np.array(boxes))
        labels = torch.tensor(np.array(labels))
        if self.img_aug != None:
            transformed = self.img_aug(image=np.asarray(image), bboxes=boxes, class_labels=labels)
                              
            augmented_image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            # for i, box in enumerate(boxes):
            #     boxes[i] = [box[0] / augmented_image.shape[0], box[1] / augmented_image.shape[1], (box[0] + box[2]) / augmented_image.shape[0], (box[1] + box[3]) / augmented_image.shape[1]]

            # transformed['class_labels']
            
            # boxes, labels = self.encoder.encode(torch.tensor(np.array(boxes), dtype = torch.float), torch.tensor(transformed['class_labels']).long())            
            image = transform(augmented_image) # image to Tensor form

        return (image, target[0]["image_id"], (augmented_image.shape[1], augmented_image.shape[1]), boxes, labels)
