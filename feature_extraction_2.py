import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os

import torch
import torch.nn as nn


import pretrainedmodels
from pretrainedmodels import utils
from collections import namedtuple

import Inception
from Inception import *
import warnings


def extract_features(frame_steps, model_name, video_path, output_dir, data_dir, model, load_image_fn):
    model.eval()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print("save video feats to %s" % (output_dir))

    image_list = sorted(glob.glob(os.path.join(data_dir)))
    for image in tqdm(image_list):
        video_id = image.split("/")[-1].split(".")[0]
    samples = np.round(np.linspace(
        0, len(image_list) - 1, frame_steps))
    image_list = [image_list[int(sample)] for sample in samples]
    images = torch.zeros((len(image_list), 3, 299, 299))
    for iImg in range(len(image_list)):
        img = load_image_fn(image_list[iImg])
        images[iImg] = img
    with torch.no_grad():
        fc_feats = model(images.cuda()).squeeze()
    img_feats = fc_feats.cpu().numpy()
    # Save the inception features
    outfile = os.path.join(output_dir, video_id + '.pt')
    np.save(outfile, img_feats)
    # cleanup
    shutil.rmtree(data_dir)


if __name__ == '__main__':
    data_dir = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\Frames\\train\\1'
    output_dir = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\results'

    model_name = 'inception_v3'
    frame_steps = 40

    # check for cuda and set gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    model = pretrainedmodels.inceptionv3(pretrained='imagenet')
    # model = Inception.inception_v3(pretrained=False, progress=True)
    load_image_fn = utils.LoadTransformImage(model)

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)

    model = model.cuda()

    extract_features(frame_steps, model_name, data_dir, output_dir, model, load_image_fn)
