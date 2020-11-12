# This script is based on a kernel found on kaggle
# The author of this tutorial is SentDex
# The link to this tutorial is here:
# https://www.kaggle.com/sentdex/data-science-bowl-2017/
# first-pass-through-data-w-3d-convnet
# This is a change in the code

#imports
import dicom  # for reading dicom files
import os  # for doing directory operations
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import csv

# These are the dimensions of our image
# The IMG_SIZE_PX represents a 50x50 image
# the slice count is the number of slices we will combine
# to make a chunk during pre-processing.
IMG_SIZE_PX = 50
SLICE_COUNT = 20

##### To understand what it means ####
batch_size = 10
IMG_PX_SIZE = 150
HM_SLICES = 20

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link:
    # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    count = 0
    for i in range(0, len(l), n):
        if(count < HM_SLICES):
            yield l[i:i + n]
            count = count + 1


def mean(l):
    return sum(l) / len(l)


def process_data(patient, img_px_size=50, hm_slices=10, visualize=False):

    path = data_dir + '/' + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),
                         (img_px_size, img_px_size)) for each_slice in slices]

    chunk_sizes = int(math.floor(len(slices) / hm_slices))
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices + 2:
        new_val = list(
            map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if len(new_slices) == hm_slices + 1:
        new_val = list(
            map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num + 1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    return np.array(new_slices)

#                                               stage 1 for real.
basepath = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(basepath, "/scratch/CS640/input/stage1/"))
# data_dir = '../Sample_images/'
patients = os.listdir(data_dir)
file_path = os.path.abspath(os.path.join(basepath, "stage1_labels.csv"))
labels = pd.read_csv(file_path, index_col=0)
train = set(labels.index)
file_path = os.path.abspath(os.path.join(basepath, "stage1_sample_submission.csv"))
test = pd.read_csv(file_path, index_col=0)
test = set(test.index)


train_data = []
test_data = []
for num, patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:	
		img_data = process_data(patient, img_px_size=IMG_SIZE_PX, 
								hm_slices=SLICE_COUNT)
		
		if patient in train:
			label = labels.get_value(patient, 'cancer')
			if label == 1:
				label = np.array([0, 1])
			elif label == 0:
				label = np.array([1, 0])
			train_data.append([img_data, label])
		if patient in test:
			test_data.append([img_data, patient])		
    except KeyError as e:
		print("key error ", patient)
		continue

np.save('traindata-{}-{}-{}.npy'.format(IMG_SIZE_PX,
                                       IMG_SIZE_PX, SLICE_COUNT), train_data)
									  

np.save('testdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,
                                       IMG_SIZE_PX, SLICE_COUNT), test_data)