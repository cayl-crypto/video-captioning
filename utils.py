import os
import sys
import math
import cv2
from tqdm import tqdm
import requests
from zipfile import ZipFile
import json
from PIL import Image
import matplotlib.pyplot as plt
import tarfile


######### Download Dataset and Annotations ###########

def download_dataset(thetarfile, target_path):
    print("Downloading...")

    ftpstream = requests.get(thetarfile, stream=True)
    total_size_in_bytes = int(ftpstream.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as f:
        for data in ftpstream.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print("Extracting file...")
    thetar = tarfile.open(target_path)
    thetar.extractall()
    thetar.close()
    print("Done.")


def download_annotations(annotation_file_test, annotation_file_train, annotation_file_val, target_path_test,
                         target_path_train, target_path_val):
    print("Annotations are downloading...")
    annotation = [annotation_file_test, annotation_file_train, annotation_file_val]
    target = [target_path_test, target_path_train, target_path_val]
    for i in range(len(annotation)) and range(len(target)):

        # Define the remote file to retrieve
        remote_url = '%s' % (annotation[i])
        # Define the local filename to save data
        local_file = '%s' % (target[i])
        # Make http request for remote file data
        getdata = requests.get(remote_url)
        total_size_in_bytes = int(getdata.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        # Save file data to local copy
        with open(local_file, 'wb')as file:
            for data in getdata.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
    print('Done.')


###########################################################################################

####### Video to Frame #########

def video_to_frame(file_paths, file_names):
    path_to_save = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\Frames\\train\\1'
    # get file path for desired video and where to save frames locally
    for file_path, file_name in tqdm(zip(file_paths, file_names)):
        raise NotImplementedError

    for i in range(len(file_paths)) and range(len(file_names)):
        cap = cv2.VideoCapture('%s' % (file_paths[i]))
        # Videos framerate cap.get(propertyId)
        frameRate = cap.get(5)  # frame rate
        x = 1

        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            # capture each frame
            ret, frame = cap.read()

            if not ret:
                break

            if frameId % math.floor(frameRate) == 0:
                # Save frame as a jpg file
                name = '%s_Frame_' % (file_names[i]) + str(int(x)) + '.jpg';
                x += 1
                print('Creating: ' + name)
                cv2.imwrite(os.path.join(path_to_save, name), frame)

        # release capture
        cap.release()
    print('Done.')


import os


def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.
    file_names = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

            file = os.path.basename(filename).split('.', 1)[0]
            file_names.append(file)  # Add it to the list.
    # print(file_paths)
    # print(file_names)
    video_to_frame(file_paths, file_names)
    return file_paths, file_names  # Self-explanatory.


######################################################################################

def load_image(path):
    # reads the image with given path

    return Image.open(path)


def show_image(img):
    # shows the given image
    img.show()


def gray_to_rgb(img):
    return img.convert('RGB')
