import torch
import cv2
import numpy as np
import os
import glob as glob
import json
from collections import defaultdict

from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)

from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform


# Read annotations
anns_file_path = 'resized_data/annotations.json'
f=open(anns_file_path, 'r',encoding="utf8")
dataset = json.loads(f.read())
f.close()

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']

#mapping images to bboxes and labels
img_to_bboxes=defaultdict(list)
img_to_labels=defaultdict(list)

for a in anns:
    img_to_bboxes[a['image_id']].append(a['bbox'])
    img_to_labels[a['image_id']].append(categories[a['category_id']]['supercategory'])



for key,value in img_to_labels.items():
    for i,j in enumerate(value):
        if j=='Paper bag' or j=='Scrap metal' or j=='Rope & strings' or j=='Food waste' or j=='Blister pack' or j=='Shoe' or j=='Squeezable tube' or j=='Glass jar' or j=='Plastic glooves' or j=='Battery' or j=='Unlabeled litter':
            img_to_labels[key][i]='Others'
                      

#The dataset class
class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images.sort(key=lambda x: int(x.split('.')[0]))
        
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        r=image_name.index('.')
        
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # box coordinates are extracted and corrected for image size given
        boxes=[]
        for j in img_to_bboxes[int(image_name[0:r])]:
            boxes.append(j)
        
        labels=[]
        for i in img_to_labels[int(image_name[0:r])]:
            labels.append(self.classes.index(i))
            
        for j in boxes:
            j[0] = (j[0]/image_width)*self.width
            j[2] = (j[2]/image_width)*self.width
            j[1] = (j[1]/image_height)*self.height
            j[3] = (j[3]/image_height)*self.height

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)
        
# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO[0], RESIZE_TO[1], CLASSES, get_train_transform())
    return train_dataset

def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO[0], RESIZE_TO[1], CLASSES, get_valid_transform())
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

#Visualize sample images from terminal
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO[0], RESIZE_TO[1], CLASSES
    )
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(800,800+NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)

        

