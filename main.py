#import torch

# TODO: Choose dataset and download using python scripts.
from models import *
import torch
from tokenization import *
import time
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_dataset(ids,captions):
    video_path="YouTubeClips"
    file_names = []  # List which will store all of the full filepaths.
    for root, directories, files in os.walk(video_path):
        for filename in files:
            file = os.path.basename(filename).split('.', 1)[0]
            file_names.append(file)  # Add it to the list.

    a=int(len(file_names)*0.70)
    b=int(len(file_names) * 0.20)
    train_captions=[]
    train_id=[]
    test_captions=[]
    test_id=[]
    val_captions=[]
    val_id=[]
    for i,cap in enumerate(file_names):
        for j,idd in enumerate(ids):
            if (idd == cap):
                if i <= a:
                    train_captions.append(captions[j])     #TRAIN
                    train_id.append(idd)
                elif i>a and i<=(a+b):
                    test_captions.append(captions[j])      #TEST
                    test_id.append(idd)
                else:
                    val_captions.append(captions[j])       #VALIDATION
                    val_id.append(idd)

            else:
                continue


    return train_captions,train_id,test_captions,test_id,val_captions,val_id


def train_tokenize(train_captions,train_id):
    voc = Voc()
    for caption in train_captions:
        voc.addCaption(caption)

    voc.trim(20)

    voc.save_train_vocabulary()
    voc.load_train_vocabulary()
    voc_size = len(voc.index2word)

    tokenized_captions = tokenize(voc, train_captions)
    # print(tokenized_captions)
    id, caption_tokens = dataset_annotation(train_id, tokenized_captions)

    padded_tokens = pad_sequences(caption_tokens)

    return id, padded_tokens,voc_size


def test_tokenize(test_captions,test_id):
    voc = Voc()
    for caption in test_captions:
        voc.addCaption(caption)

    voc.trim(20)

    voc.save_test_vocabulary()
    voc.load_test_vocabulary()
    voc_size = len(voc.index2word)

    tokenized_captions = tokenize(voc, test_captions)
    # print(tokenized_captions)
    id, caption_tokens = dataset_annotation(test_id, tokenized_captions)

    padded_tokens = pad_sequences(caption_tokens)

    return id, padded_tokens,voc_size


def val_tokenize(val_captions,val_id):
    voc = Voc()
    for caption in val_captions:
        voc.addCaption(caption)

    voc.trim(20)

    voc.save_val_vocabulary()
    voc.load_val_vocabulary()
    voc_size = len(voc.index2word)

    tokenized_captions = tokenize(voc, val_captions)
    # print(tokenized_captions)
    id, caption_tokens = dataset_annotation(val_id, tokenized_captions)

    padded_tokens = pad_sequences(caption_tokens)

    return id, padded_tokens,voc_size



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

train_captions,train_id,test_captions,test_id,val_captions,val_id=split_dataset(ids,captions)
id, padded_tokens,voc_size=train_tokenize(train_captions,train_id)


train_tokens = torch.LongTensor(padded_tokens).to(device)


vall_id, val_padded_tokens, val_voc_size = val_tokenize(val_captions, val_id)
val_tokens = torch.LongTensor(val_padded_tokens).to(device)
batch_size=64


def generate_caption(encoder,decoder, video_frames, max_len=20):

    voc = Voc()

    voc.load_train_vocabulary()
    encoder_hidden = torch.zeros(1, 1, hidden_size).to(device)
    input_length = video_frames.size(1)
    with torch.no_grad():
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder.forward(
                (video_frames[0,ei,:].unsqueeze(0)).unsqueeze(0), encoder_hidden)

    decoder_hidden = encoder_hidden
    input_token=torch.ones(1,1).type(torch.LongTensor).to(device)
    caption = ""

    for seq in range(max_len):

        #input_token = torch.ones(1,1).type(torch.LongTensor).to(device)
        #input_token = input_token.unsqueeze(0)

        with torch.no_grad():
            decoder_output, decoder_hidden = decoder(input_token, decoder_hidden)
        decoder_output = decoder_output.argmax(dim=1)
        caption += voc.index2word[str(int(decoder_output))] + " "
        input_token = decoder_output.unsqueeze(0)

    print(caption)

def train(video_frames,target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    #encoder_hidden = encoder.initHidden()
    encoder_hidden=torch.zeros(1,video_frames.size(0),hidden_size).to(device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = video_frames.size(1)
    #target_length = caption.size(0)
    sequence_length = target_tensor.size(1)
    #batch_size = caption.size(1)

    #encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder.forward(
            video_frames[:,ei,:].unsqueeze(0), encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    for seq in range(sequence_length - 1):
        input = target_tensor[:,seq].long()
        input = input.unsqueeze(0)
        output = target_tensor[:,seq + 1].long()

        decoder_output, decoder_hidden = decoder(input, decoder_hidden)
        loss += criterion(decoder_output, output)


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / sequence_length

def val(video_frames,target_tensor, encoder, decoder,criterion):
    encoder_hidden = torch.zeros(1, video_frames.size(0), hidden_size).to(device)

    input_length = video_frames.size(1)
    # target_length = caption.size(0)
    sequence_length = target_tensor.size(1)
    # batch_size = caption.size(1)

    # encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0
    with torch.no_grad():
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder.forward(
                video_frames[:, ei, :].unsqueeze(0), encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    for seq in range(sequence_length - 1):
        input = target_tensor[:, seq].long()
        input = input.unsqueeze(0)
        output = target_tensor[:, seq + 1].long()
        with torch.no_grad():
            decoder_output, decoder_hidden = decoder(input, decoder_hidden)
        loss += criterion(decoder_output, output)


    return loss.item() / sequence_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    video_frames = torch.zeros(batch_size, 8, 2048).to(device)
    target_tensor = torch.zeros(batch_size, train_tokens.shape[1]).to(device)
    counter=0
    best_val_loss=None
    STOP_POINT=10
    #generate_caption(encoder=encoder, decoder=decoder, video_frames=video_frames)
    for iters in tqdm(range(1, n_iters + 1)):
        encoder.train()
        decoder.train()
        for n, index in enumerate(id):
            batch_index = n % batch_size
            video_frames[batch_index] = torch.load('features/' + index + '.pt')  # encoder input
            target_tensor[batch_index] = train_tokens[n]
            if batch_index == batch_size - 1:
                loss = train(video_frames, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)
                print_loss_total += loss
                plot_loss_total += loss

            if n == len(id) - 1:
                loss = train(video_frames[:batch_index+1], target_tensor[:batch_index+1], encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)
                print_loss_total += loss
                plot_loss_total += loss


        #if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print(f"train loss: {print_loss_avg}")
        print()

        print_val_loss_total = 0  # Reset every print_every
        plot_val_loss_total = 0
        encoder.eval()
        decoder.eval()

        for n, index in enumerate(vall_id):
            batch_index = n % batch_size
            video_frames[batch_index] = torch.load('features/' + index + '.pt')  # encoder input
            target_tensor[batch_index] = val_tokens[n]

            if batch_index == batch_size - 1:


                val_loss =val(video_frames,target_tensor, encoder, decoder,criterion)

                print_val_loss_total += val_loss
                plot_val_loss_total += val_loss

            if n == len(vall_id) - 1:

                val_loss = val(video_frames[:batch_index+1],target_tensor[:batch_index+1], encoder, decoder,criterion)
                print_val_loss_total += val_loss
                plot_val_loss_total += val_loss

        print_val_loss_avg = print_val_loss_total / print_every
        print_val_loss_total = 0
        print(f"validation loss: {print_val_loss_avg}")
        print()
        if best_val_loss==None:
            best_val_loss=print_val_loss_avg
            torch.save(encoder, 'best_encoder.pth')
            torch.save(decoder, 'best_decoder.pth')
        elif print_val_loss_avg< best_val_loss:
            counter=0
            best_val_loss=print_val_loss_avg
            torch.save(encoder, 'best_encoder.pth')
            torch.save(decoder, 'best_decoder.pth')
        else:
            counter+=1
            if counter>=STOP_POINT:
                break
        generate_caption(encoder=encoder,decoder=decoder, video_frames=video_frames)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)




feature_size=2048
hidden_size = 256
num_layers = 1
n_iters=20
output_size=voc_size
encoder1 = EncoderRNN(feature_size, hidden_size).to(device)
decoder=DecoderRNN(hidden_size,output_size).to(device)
trainIters(encoder1,decoder,n_iters)

















