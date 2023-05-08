import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#class weights
class_weights=torch.FloatTensor([1.0,3.3363,0.5577,0.9114,2.2242,0.8570,1.1056,1.3157,2.919,0.9484,1.7461,0.3205,3.0628,8.492,2.631,2.4264,3.2213,0.3579,0.4766])


def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    
    
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    #freeze the backbone (it will freeze the body and fpn params)
    for p in model.backbone.parameters():
        p.requires_grad=False

    #freeze the fc6 layer in roi_heads
    for p in model.roi_heads.box_head.fc6.parameters():
        p.requires_grad=False
    
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_predictor.cls_loss_func = torch.nn.CrossEntropyLoss(weight=class_weights,label_smoothing=0.1)

    return model

