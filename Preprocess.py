import os
import cv2
import numpy as np
from PIL import Image
from skimage.io import imsave
from facenet_pytorch import MTCNN

# this is the MTCNN face detector with basic parameters
mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device="cuda:0")

# this is the source folder path and destination folder path !!! Change these paths to where you installed the FaceForensics++ database!!!
source_folder_path = "C:/Users/Jeet/Documents/FF++"
destination_folder_path = "C:/Users/Jeet/Documents/FF++/train_faces"


# looping in source folder
for root, dirs, files in os.walk(source_folder_path):
    for vn in files:
        vp = os.path.join(root, vn)
        r = os.path.relpath(source_folder_path, root)
        new_face_folder = os.path.join(destination_folder_path, r, os.path.splitext(vn)[0])
        # making new folder
        if not os.path.exists(new_face_folder):
            os.makedirs(new_face_folder)
        # reading video capture in video path
        vi_cap = cv2.VideoCapture(vp)
        # getting frame rate
        frame_r = vi_cap.get(cv2.CAP_PROP_FPS)
        skip = int(frame_r * 10)
        f_count = 0
        flag = True

        # looping frame index in video capture count
        for i in range(0, int(vi_cap.get(cv2.CAP_PROP_FRAME_COUNT)), skip):
            flag, frame = vi_cap.read()
            if not flag:
                break
            # converting to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f_image = Image.fromarray(rgb_frame)
            # creating MTCNN
            f = mtcnn(f_image)
            if f is not None:
                f_image = f.permute(1, 2, 0).numpy().astype(np.uint8)
                f_path = os.path.join(new_face_folder, f"frame_{f_count}.jpg")
                # saving image
                imsave(f_path, f_image)
                f_count += 1
            else:
                print("There is no face detected.")
        vi_cap.release()
        print(f"Processed video: {vn}")

