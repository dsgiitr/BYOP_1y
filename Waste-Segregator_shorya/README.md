
# Waste Segregator

### Project Description

The project uses object detection to detect waste in the wild, the model uses a pretrained faster rcnn resnet50 and is trained on the [taco dataset](http://tacodataset.org/).

### Prerequisites

You can install the libraries needed using
```
pip install -r requirements.txt
```
The python version used in the project is 3.10.11

### Installation

To clone the git repository use
```
git clone https://github.com/Shorya1835/Waste-Segregator.git
```


### Libraries Used

 - Pytorch
   - The model uses torch, torchvision for the model and torchmetrics to evaluate the model.
- Numpy
     - For arrays
- Cv2 and matplotlib
    - For visualization
- albumentaions
    - For transforming the data
- glob
    - To search for files 
- Chitra and ppyboxes
    - During data analysis for data conversion and plotting

### Usage

The model can detect waste in the wild and categorize it between 18 categories, which can be furhter mapped to recylable and organic, allowing effective waste segregation. The model has mAP of 6 percent, with highest 7 percent achieved.

![An example](https://i.imgur.com/axekldV.png)

### Structure 

```
Faster_rcnn_implementation
├── config.py
├── custom_utils.py
├── datasets.py
├── image_distributor.ipynb
├── inference.py
├── model.py
├── outputs
│   ├── best_model.pth
│   ├── train_loss.png
│   └── valid_loss.png
├── resized_data
│   ├── annotations.json
│   ├── test
│   ├── train
│   └── validation
└── train.py
```
