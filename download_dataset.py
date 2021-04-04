import os
import sys
from tqdm import tqdm
import requests
from zipfile import ZipFile
import json
from PIL import Image
import matplotlib.pyplot as plt
import tarfile
from utils import *

def Download_Dataset():
    thetarfile = "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
    target_path = "C:\\Users\\pc\\PycharmProjects\\video-captioning\\YouTubeClips.tar"
    # download_dataset(thetarfile,target_path)

def Download_Annotations():
    # Please add 'annotations' folder to file path before running code.

    annotation_file_test = "https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAAfs3zbjpeYtzfOOeFzdPMta/sents_test_lc_nopunc.txt"
    annotation_file_train = "https://www.dropbox.com/sh/4ecwl7zdha60xqo/AACLdedalP2OIPu5KG6cg5G7a/sents_train_lc_nopunc.txt"
    annotation_file_val = "https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAAU2dioWf_vRTW2Gqgnd4b5a/sents_val_lc_nopunc.txt"
    target_path_test = "/annotations\\sents_test_lc_nopunc.txt"
    target_path_train = "/annotations\\sents_train_lc_nopunc.txt"
    target_path_val = "/annotations\\sents_val_lc_nopunc.txt"
    #download_annotations(annotation_file_test, annotation_file_train, annotation_file_val, target_path_test,
    #                     target_path_train, target_path_val)



def main():

      Download_Dataset()
      Download_Annotations()


if __name__ == "__main__":
    main()