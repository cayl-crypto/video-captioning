
import csv
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import unicodedata
import re
from tqdm import tqdm
from create_vocabulary import *

class Vocabulary:

    def __init__(self):
        self.train_list = []
        self.vocab_list = []

        self.word = Counter()

    def caption_tokens(self,annotation,vidid,vocab_file):
        # train_list contains all the captions with their video ID
        # vocab_list contains all the vocabulary from training data


        for caption,vid_id in zip(annotation,vidid):
            caption = ' '.join(word.strip(string.punctuation).lower() for word in caption.split())
            caption = "<bos> " + caption + " <eos>"

            self.train_list.append([caption, vid_id])

        for i in annotation:

            x=word_tokenize(i)
            self.word += Counter(x)

            for j in x:
               self.vocab_list.append(j)

        #print(self.word)
        vocab_list=set(self.vocab_list)
        vocab_list=sorted(vocab_list)
        #print(vocab_list)

        word=self.word
        for key, f in sorted(word.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write(key + " " + str(f) + "\n")

        with open('vocabulary.csv', 'a') as csvfile:
           writer = csv.writer(csvfile, dialect='excel')

           for k in vocab_list:
                writer.writerow([k])


def main():

    """
    vidid=[]
    annotation=[]
    with open('annotationswithid.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            vidid.append(row[0])
            annotation.append(row[1])

    """

    fl = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\annotation with id\\sentences.txt'
    fileObj = open(fl, "r") #opens the file in read mode
    annotation= fileObj.read().splitlines()

    fle = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\annotation with id\\id.txt'
    fileObj = open(fle, "r") #opens the file in read mode
    vidid= fileObj.read().splitlines()

    vocab_file = open("vocab_file.txt", "w+")

    vocab=Vocabulary()
    vocab.caption_tokens(annotation,vidid,vocab_file)

    annotation=normalizeAllCaptions(annotation)
    vocab = Voc()
    for caption in annotation:
        vocab.addCaption(caption)

    vocab.trim(20)

    vocab.save_vocabulary()

if __name__ == "__main__":
    main()