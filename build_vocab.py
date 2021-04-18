
import csv
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import unicodedata
import re
from tqdm import tqdm
from create_vocabulary import *
from create_json_annotation_id import *

class Vocabulary:

    def __init__(self):
        self.train_list = []
        self.vocab_list = []

        self.word = Counter()

    def new_caption(self, annotation, vidid):
        # train_list contains all the captions with their video ID
        # vocab_list contains all the vocabulary from training data
        new_annotation=[]
        for caption,vid_id in zip(annotation,vidid):
            caption = ' '.join(word.strip(string.punctuation).lower() for word in caption.split())
            new_caption = "bos " + caption + " eos"
            new_annotation.append(new_caption)
            #self.train_list.append(caption)
        return new_annotation
    def caption_tokens(self, annotation, vidid, vocab_file):
        for i in annotation:

            x=word_tokenize(i)
            self.word += Counter(x)

            for j in x:
               self.vocab_list.append(j)

        #print(self.word)
        vocab_list=set(self.vocab_list)
        vocab_list=sorted(vocab_list)
        #print(vocab_list)

        """
        word=self.word
        for key, f in sorted(word.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write(key + " " + str(f) + "\n")

        with open('vocabulary.csv', 'a') as csvfile:
           writer = csv.writer(csvfile, dialect='excel')

           for k in vocab_list:
                writer.writerow([k])

        """
def main():

    file_path = "annotation with id\\video_captioning.json"
    with open(file_path) as f:
        annotations = json.load(f)
    captions = []
    ids = []
    for annotation in tqdm(annotations):
        caption = annotation['caption']
        id = annotation['id']
        captions.append(caption)
        ids.append(id)


    vocab_file = open("vocab_file.txt", "w+")

    vocab=Vocabulary()
    vocab.caption_tokens(captions,ids,vocab_file)
    new_annotation=vocab.new_caption(captions, ids)
    captions=normalizeAllCaptions(new_annotation)

    output_file = "video_captioning_normalized.json"

    crf = Annotations()
    crf.read_file(new_annotation, ids)
    crf.dump_json(output_file)

    vocab = Voc()
    for caption in captions:
        vocab.addCaption(caption)

    vocab.trim(20)

    vocab.save_vocabulary()

if __name__ == "__main__":
    main()