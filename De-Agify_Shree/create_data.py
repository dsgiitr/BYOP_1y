import tensorflow as tf
import numpy as np
import os
import random
from joblib import load, dump
from matplotlib import pyplot as plt

shape = 128
AGE_MEAN, AGE_STD = 35.3, 15.54

divs = 2

files = os.listdir('UTKFace')
# file name is [age]_[gender]_[race]_[date&time].jpg

# we set a lower bound for age
files = list(filter(lambda x:6<int(x.split('_')[0]), files))
ages = np.array([int(x.split('_')[0]) for x in files])
genders = np.array([int(x.split('_')[1]) for x in files])

random.shuffle(files)

set_size = len(files)//divs
for i in range(divs):
    
    file_set = files[i*set_size:(i+1)*set_size]
    if i == divs-1:
        file_set = files[i*set_size:]
        
    ages = (np.array([int(x.split('_')[0]) for x in file_set], dtype='float32')-AGE_MEAN)/AGE_STD
    genders = np.array([int(x.split('_')[1]) for x in file_set])
    
    images = np.zeros((len(file_set), shape, shape, 3), dtype='float32')

    for j, file in enumerate(file_set):
        images[j] = tf.image.resize(tf.keras.utils.img_to_array(tf.keras.utils.load_img(f"UTKFace/{file}")), (shape, shape))

    images = (images-127.5)/127.5
    dump(images, f'data/{i}_images.joblib')
    dump(ages, f'data/{i}_ages.joblib')
    dump(genders, f'data/{i}_genders.joblib')
    
    plt.hist(ages, bins=30)
    plt.show()
    plt.hist(genders)
    plt.show()

