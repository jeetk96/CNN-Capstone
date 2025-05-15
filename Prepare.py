import cv2
import numpy as np
import glob
import os
from os import listdir
from os.path import join
import argparse

# this is for argument parsing
argument_parsing = argparse.ArgumentParser()
argument_parsing.add_argument(
    "-img_size",
    "--img_size",
    type=int,
    help="Resize face image",
    default=160,
)
argument_parsing.add_argument(
    "-fpv",
    "--frames_per_video",
    type=int,
    help="Number of frames per video to consider",
    default=25,
)
args = argument_parsing.parse_args()

# this is the path of train_faces folder !!! Change these paths to where you installed the FaceForensics++ database!!!
train_path = "C:/Users/Jeet/Documents/FF++/train_faces"

# face images paths and labels
face_paths = []
labels = []

# this is looping with 'real' and 'fake'
for lbl in ['real', 'fake']:
    fp = join(train_path, lbl)
    if not os.path.exists(fp):
        continue
    for vf in listdir(fp):
        vp = join(fp, vf)
        if not os.path.isdir(vp):
            continue
        # this is generating jpg images in video folder
        jpg_imgs = glob.glob(join(vp, "*.jpg"))
        jpg_imgs.sort()
        face_paths.extend(jpg_imgs[:args.frames_per_video])
        labels.extend([lbl] * min(len(jpg_imgs), args.frames_per_video))

# checking if there are any images in face_paths
from random import shuffle
if not face_paths:
    print("There are no images in face_paths")
else:
    # shuffle data
    combined = list(zip(face_paths, labels))
    shuffle(combined)
    face_paths, labels = zip(*combined)

# training data and labels
my_train_data = []
my_train_labels = []

for img_path, label in zip(face_paths, labels):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Unable to read image {img_path}.")
        continue  # skip this status
    # convert image to RGB color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize the image
    img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
    # putting the image and label into train_data and train_labels
    my_train_data.append(img)
    my_train_labels.append(label)

# convert lists to numpy arrays
my_train_data = np.array(my_train_data)
my_train_labels = np.array(my_train_labels)

# this assigns a label to the image with 1 if real and 0 if fake
from keras.utils import to_categorical
my_train_labels = to_categorical(np.where(my_train_labels == 'real', 1, 0))

# saving the processed data
np.save("train_data.npy", my_train_data)
np.save("train_labels.npy", my_train_labels)
