import os
import glob
import cv2
import time
import random
import shutil
import numpy as np
from keras.models import model_from_json
from utils_gen import gen_paths_img_dm, gen_var_from_paths
from utils_imgproc import norm_by_imagenet


# Settings
net = 'CSRNet'

def run(image_file, part_name):
    # Checking paths if contains unseen images, whether belong to A or B folder:
    root = 'part/' + part_name

    # if len(os.listdir(root +'A')) == 0:
    #     root=root+'B'
    # else:
    #     root=root +'A'

    # create paths from selected folder (A or B)
    img_new_paths=[]
    for img_path in glob.glob(os.path.join('part', part_name, image_file.name)):
        img_new_paths.append(img_path)

    # Generate raw images(normalized by imagenet rgb) and density maps
    test_x= gen_var_from_paths(img_new_paths[:], unit_len=None)
    test_x = norm_by_imagenet(test_x)  # Normalization on raw images in test set, those of training set are in image_preprocessing below.
    print('Test data size:', test_x.shape[0], len(img_new_paths))

    # Analysis on results
    dis_idx = 16 if root=='part/B' else 0 # I UNCOMMENTED THIS ONE AS THIS SHOULD LOAD THE ORIGINAL BEST WEIGHTS

    # set condition to choose weight A or B
    if root=='part/B': 
        weights_dir_neo = "weights_B_MSE_bestMAE8.31_Sun-May-19" # Best weights for set B
    else:
        weights_dir_neo = "weights_A_MSE_bestMAE67.984_Thu-May-23" # Best weights for set A


    model = model_from_json(open('models/{}.json'.format(net), 'r').read())  # I UNCOMMENTED THIS ONE AS THIS SHOULD LOAD THE ORIGINAL BEST WEIGHTS
    model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))

    ct_preds = []
    ct_gts = []
    for i in range(len(test_x[:])):
        if i % 100 == 0:
            print('{}/{}'.format(i, len(test_x)))
        i += 0
        test_x_display = np.squeeze(test_x[i])

        pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
        ct_pred = np.sum(pred)

        ct_preds.append(ct_pred)
       
    for i in range(len(ct_preds)):
        return round(ct_preds[i])
