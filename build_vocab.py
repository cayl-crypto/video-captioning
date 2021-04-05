

import os
import numpy as np
from tokenizers import Tokenizer
import csv


def caption_tokens():
    # train_list contains all the captions with their video ID
    # vocab_list contains all the vocabulary from training data
    train_list = []
    vocab_list = []
    for caption,vid_id in zip(y_data,vidid):
        caption = "<bos> " + caption + " <eos>"
        # we are only using sentences whose length lie between 6 and 10
        if len(caption.split())>10 or len(caption.split())<6:
          continue
        else:
          train_list.append([caption, vid_id])
    print(len(train_list))

    print(train_list)
    for train in train_list:
        vocab_list.append(train[0])
    #print(vocab_list)
    # Tokenizing the words
    tokenizer = Tokenizer(num_words=9462)
    tokenizer.fit_on_texts(vocab_list)
    print(tokenizer)
    """
    x_data = {}
    TRAIN_FEATURE_DIR = os.path.join('training_data', 'feat')
    # Loading all the numpy arrays at once and saving them in a dictionary
    for filename in os.listdir(TRAIN_FEATURE_DIR):
        f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename))
        x_data[filename[:-4]] = f
    """

vid_id_path="vid_id.txt"
fileName="sentences.txt"
fileObj = open(fileName, "r") #opens the file in read mode
y_data = fileObj.read().splitlines()
vidfile = open(vid_id_path, "r") #opens the file in read mode
vidid = vidfile.read().splitlines()
caption_tokens()
