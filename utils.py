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
import os
import datetime
import csv
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
        # Define the local filename to save annotations
        local_file = '%s' % (target[i])
        # Make http request for remote file annotations
        getdata = requests.get(remote_url)
        total_size_in_bytes = int(getdata.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        # Save file annotations to local copy
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

def video_to_frame(file_paths, file_names,path_to_save):
    os.makedirs(path_to_save)
    # get file path for desired video and where to save frames locally

    file_paths.sort()
    file_names.sort()
    #video_time_path = open("video_times.txt", "w+")
    """
    with open('Frame_FPS.csv', mode='w') as new_file:
        new_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        new_writer.writerow(['Video ID','Frames', 'FPS'])
        """
    for file_path, file_name in tqdm(zip(file_paths, file_names)):

        cap = cv2.VideoCapture(file_path)
        # count the number of frames
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        #new_writer.writerow((file_name,frames,fps))

        #fpsnumber.write(frames+"     "+fps + "\n")
        # calculate duration of the video
        seconds = int(frames / fps)
        video_time = str(datetime.timedelta(seconds=seconds))
        #print("video time:", video_time)
        #video_time_path.write(file_name +"  "+"--->"+"  "+ video_time+"\n")
        # Videos framerate cap.get(propertyId)
        #frameRate = cap.get(5)  # frame rate
        frameRate=math.ceil(frames/8)
        x = 1

        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            # capture each frame
            ret, frame = cap.read()

            if not ret:
                break

            if frameId % math.floor(frameRate) == 0:
                # Save frame as a jpg file
                name = '%s_Frame_' % (file_name) + str(int(x)) + '.jpg';
                x += 1
                print('Creating: ' + name)
                cv2.imwrite(os.path.join(path_to_save, name), frame)

        # release capture
        cap.release()

    #video_time_path.close()

    print('Done.')





def get_filepaths(video_path,path_to_save):
    file_paths = []  # List which will store all of the full filepaths.
    file_names = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(video_path):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

            file = os.path.basename(filename).split('.', 1)[0]
            file_names.append(file)  # Add it to the list.
    # print(file_paths)
    # print(file_names)
    video_to_frame(file_paths, file_names,path_to_save)

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
