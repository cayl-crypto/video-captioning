import os
import sys
from tqdm import tqdm
import requests
from zipfile import ZipFile
import json
from PIL import Image
import matplotlib.pyplot as plt
import tarfile

def download_dataset(thetarfile,target_path):
    print("Downloading...")

    ftpstream = requests.get(thetarfile,stream=True)
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



"""
def load_image(path):
    # reads the image with given path

    return Image.open(path)


def show_image(img):
    # shows the given image
    img.show()


def gray_to_rgb(img):
    return img.convert('RGB')

"""

def main():
    thetarfile = "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
    target_path= "C:\\Users\\pc\\PycharmProjects\\video-captioning\\YouTubeClips.tar"
    download_dataset(thetarfile,target_path)


if __name__ == "__main__":
    main()