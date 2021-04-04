import cv2
import numpy as np
import os
import math
from utils import *


## ADD 'Frames' folder to file path before running code.

def Video_to_Frames():
    # Run the above function and store its results in a variable.
    video_path="C:\\Users\\pc\\PycharmProjects\\video-captioning\\YouTubeClips"
    path_to_save = "C:\\Users\\pc\\PycharmProjects\\video-captioning\\Frames\\train\\"

    get_filepaths(video_path,path_to_save)


def main():
    Video_to_Frames()


if __name__ == "__main__":
    main()
