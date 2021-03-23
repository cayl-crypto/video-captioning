import os
import sys
from tqdm import tqdm
import requests
from zipfile import ZipFile
import json
from PIL import Image
import matplotlib.pyplot as plt
import tarfile


def load_image(path):
    # reads the image with given path

    return Image.open(path)


def show_image(img):
    # shows the given image
    img.show()


def gray_to_rgb(img):
    return img.convert('RGB')



