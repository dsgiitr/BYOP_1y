import torch

#Hyperparameters
BATCH_SIZE=7
RESIZE_TO=(256,224) 
NUM_EPOCHS=500
NUM_WORKERS=0

#Choosing between cpu or gpu
DEVICE=torch.device(('cuda') if torch.cuda.is_available() else torch.device('cpu'))

#Training images
TRAIN_DIR='resized_data/train'

#Validation images 
VALID_DIR='resized_data/validation'

#classes: 0 index is reserved for background
CLASSES=[
    '__background__','Aluminium foil', 'Bottle', 'Bottle cap', 'Broken glass', 'Can', 'Carton', 'Cup', 'Lid', 'Other plastic', 'Paper', 'Plastic bag & wrapper', 'Plastic container', 'Plastic utensils', 'Pop tab', 'Straw', 'Styrofoam piece','Cigarette','Others']

NUM_CLASSES=len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'



    
