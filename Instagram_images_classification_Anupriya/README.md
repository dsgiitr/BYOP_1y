# Multi-Approach Image Classification
This project explores several supervised and unsupervised learning models such as k-means clustering, transfer learning on Xception, MobileNetv2 and VGG16, and sequential models created and trained from scratch, to achieve Image classification on unconventional datasets created using images extracted from Instagram and Kaggle.The end result is a comparative study between the challenges faced in implementing each model and the accuracy achieved by them on the custom made dataset.

## Datasets 
The datasets used for this project were created manually and have been uploaded on kaggle to be downloaded to train all the models and compare findings. 

- [Dataset 1](https://www.kaggle.com/datasets/anupriyakkumari/instagram-5-classes-dataset-1) - Created by manually downloading Instagram images : 50 images per class with a total of 5 classes namely animals, food, beauty, memes and travel.

- [Dataset 2](https://www.kaggle.com/datasets/anupriyakkumari/instagram-5-classes-dataset-2) - Created by merging kaggle datasets with roughly 1000-1500 images per class: 5 classes - pets (cats and dogs), food(Indian food), beauty, travel(oceans, gardens and mountains) and memes (a few popular memes).

- For the unsupervised learning model that uses k-means clustering, initially,  Instaloader API was used to scrap images from saved posts on user's Instagram account. However due to privacy issues and bugs in the API which blocked the user's IP after a few attempts, the API could no longer be reliably used. Results were produced on the first set of images that were successfully scrapped.

## Set up
The code can be run without any requirements on Google Colab. 
Download the aforementioned datasets -

1. [Dataset 1](https://www.kaggle.com/datasets/anupriyakkumari/instagram-5-classes-dataset-1)

2. [Dataset 2](https://www.kaggle.com/datasets/anupriyakkumari/instagram-5-classes-dataset-2)

The code to directly download these datasets on a colab notebook is available in `download_datasets.ipynb`(run after downloading the kaggle.json file from your kaggle account upon cretaing a new API Token) Or follow [these steps](https://www.kaggle.com/discussions/general/156610). 

Change the names of the dataset directories to "Instagram_Dataset_1" and "Instagram_Dataset_2" for consistency. The code uses these names.

User might have to change dataset directory paths in the code if the dataset is not downloaded directly on colab and loaded from google drive or some other folder instead. Rename folders accordingly.

## Reproduce findings
- Train the models by running all the cells under `Train model` in `transfer_learning_models.ipynb` and `sequential_models.ipynb` The models can be re-used on any dataset by changing the dataset directory paths. A description about the purpose and working of each subsection in the notebooks has been provided.
- Save the models by running the respective cells under `Save model` under each section for each model. The models get saved on google drive but the path can be manipulated to save them elsewhere.
- Run the cells `Test model` under each section for each model to check the accuracy of each model. Once again, paths for the unseen dataset can be changed as needed.
- The images scrapped using Instaloader API could not be obtained again. User must change the paths of the dataset in `unsupervised_learning_model.ipynb` in order to reproduce the results on their own dataset. Run all the cells in the respective sections in order to train, save and test the model.

## Findings
A comprehensive report of all the modifications made to this project, the methodology and tech stack used, key-takeaways, results and the final deliverable can be found [at this Google Doc](https://docs.google.com/document/d/1xnR8ca-tTOwe9GzMxly7aJKz1bfN-aGwtnxuXJMdI1A/edit?usp=sharing).

