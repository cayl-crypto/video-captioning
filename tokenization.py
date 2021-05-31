from tqdm import tqdm
import json
from create_vocabulary import *
from collections import Counter

def tokenize(vocabulary, captions):
    
    #print()
    #print("Tokenizing captions...")
    tokenized_captions = []
    for caption in tqdm(captions):
        caption_tokens = []
        for word in caption.split(" "):
            if word in vocabulary.word2index:
                caption_tokens.append(vocabulary.word2index[word])
           
        tokenized_captions.append(caption_tokens)
    return tokenized_captions
        

def get_maximum_length_of_captions(tokenized_captions):
    max_len = 0

    caption_len=[]
    for tokenized_caption in tqdm(tokenized_captions):
        len_cap=len(tokenized_caption)
        caption_len.append(len_cap)
        if len_cap > max_len:
            max_len = len_cap

    #print(caption_len)
    a = dict(Counter(sorted(caption_len)))
    #print(a)
    #print(max_len)

    return max_len


def pad_sequences(sequences):
    max_len = get_maximum_length_of_captions(sequences)

    padded_tokens = []

    #print("Padding tokens...")
    for sequence in tqdm(sequences):
        
        for i in range(max_len - len(sequence) + 1):
            sequence.append(0)
        padded_tokens.append(sequence)

    #print("Padded.")
    return padded_tokens

def dataset_annotation(ids, tokenized_captions):

    id = []
    caption_tokens = []
    for img_path, caption in zip(ids, tokenized_captions):
        #print(len(caption))
        if 13 >= len(caption) >= 5:

            id.append(img_path)
            caption_tokens.append(caption)

    #print("LEN of core im path")
    #print(len(id))
    #print(len(caption_tokens))
    return id, caption_tokens

def dataset_validation(ids, tokenized_captions):

    id = []
    caption_tokens = []
    for img_path, caption in zip(ids, tokenized_captions):
        #print(len(caption))
        if 5 >= len(caption) >= 1:

            id.append(img_path)
            caption_tokens.append(caption)

    #print("LEN of core im path")
    #print(len(id))
    #print(len(caption_tokens))
    return id, caption_tokens

def load_json_file(path):
    with open(path, "r") as f:
        file = json.load(f)
    return file

def main():

    file_path="video_captioning_normalized.json"
    with open(file_path) as f:
        annotations = json.load(f)
    captions = []
    ids=[]
    for annotation in tqdm(annotations):
        caption=annotation ['caption']
        id=annotation ['id']
        captions.append(caption)
        ids.append(id)

    voc = Voc()
    voc.load_vocabulary()


    tokenized_captions=tokenize(voc,captions)

    caption_tokens=dataset_annotation(ids, tokenized_captions)

    pad_sequences(caption_tokens)

    caption_val=dataset_validation(ids, tokenized_captions)
    pad_sequences(caption_val)













if __name__ == "__main__":
    main()
