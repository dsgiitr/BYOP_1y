import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
from datasets import(
    img_to_bboxes,img_to_labels
)
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision as mAP



# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
pred=defaultdict(list)

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
model.roi_heads.box_predictor.cls_loss_func = torch.nn.CrossEntropyLoss()
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# directory where all the images are present
DIR_TEST = 'resized_data/test'
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

# Create an instance of the AveragePrecision metric
ap_metric=mAP(num_labels=NUM_CLASSES)

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold =0.35

# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()
    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    #data for mAP
    pred_labels = outputs[0]['labels']
    pred_scores=outputs[0]['scores']
    pred_boxes=outputs[0]['boxes']
    gt_boxes=torch.Tensor(img_to_bboxes[int(image_name)])
    gt_labels=torch.Tensor([int(CLASSES.index(i)) for i in img_to_labels[int(image_name)]])
    # Update the metric with the predicted and ground-truth scores, labels, and boxes
    ap_metric.update([{'scores': pred_scores, 'labels': pred_labels, 'boxes': pred_boxes}],
              [{'labels': gt_labels, 'boxes': gt_boxes}])
    
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = pred_boxes.data.numpy()
        scores = pred_scores.data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        
        # get all the predicited class names
        pred_classes = [CLASSES[int(i)] for i in pred_labels.cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            pred[image_name].append([int(box[0]),int(box[1]),int(box[2]),int(box[3]),class_name])
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 2)
            cv2.putText(orig_image, class_name, 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                        2, lineType=cv2.LINE_AA)
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(0)
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)
print('TEST PREDICTIONS COMPLETE')
# Compute the mAP for all classes
mAP_all = ap_metric.compute()
map_all=mAP_all['map']
print(mAP_all)
print(f"mAP: {map_all:.4f}")
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")

