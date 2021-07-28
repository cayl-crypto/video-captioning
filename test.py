import nltk

from models import *
import torch
from tokenization import *
import time
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splt_dataset(ids,captions):
    """
    video_path="YouTubeClips"
    video_time_path = open("video_id.txt", "w+")
    file_names = []  # List which will store all of the full filepaths.
    for root, directories, files in os.walk(video_path):
        for filename in files:
            file = os.path.basename(filename).split('.', 1)[0]
            file_names.append(file)  # Add it to the list.
            video_time_path.write(file + "\n")
    video_time_path.close()
    """
    fl = 'video_id.txt'
    fileObj = open(fl, "r")  # opens the file in read mode
    file_names = fileObj.read().splitlines()
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
                    #print(i)
                    test_captions.append(captions[j])      #TEST
                    test_id.append(idd)

                else:
                    #print(i)
                    val_captions.append(captions[j])       #VALIDATION
                    val_id.append(idd)



    return train_captions,train_id,test_captions,test_id,val_captions,val_id


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

train_captions,train_id,test_captions,test_id,val_captions,val_id=splt_dataset(ids,captions)
import csv
with open('results/validation.csv', mode='w') as new_file:
        new_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        new_writer.writerow(['id','caption'])
        for z,cap in zip(val_id,val_captions):
            new_writer.writerow((z,cap))

i=0
#print(test_id)
"""
while test_id[i]==test_id[i+1]:
        #print(i)
        test_id.pop(i+1)
        if i==len(test_id)-1:
            break
        if test_id[i]!=test_id[i+1]:
            i=i+1
"""
tst_caption=[]
ref_captions = []
referance_caption=[]
for z in test_captions:
    tst_caption.append(z)
    #print(tst_caption)
    tst_caption = tst_caption[0].split()

    for i, k in enumerate(tst_caption):
        if k == 'eos':
            continue
        elif k == 'bos':
            continue
        else:
            ref_captions.append(k)

    referance_caption.append(ref_captions)
    ref_captions=[]
    tst_caption=[]
    #string = ' '.join(ref_captions)
    #ref_captions.append(string)
#print(referance_caption)
batch_size=64
hidden_size = 256


def generate_caption(encoder,decoder, video_frames,prediction, max_len=20):

    voc = Voc()
    voc.load_vocabulary()
    encoder_hidden = torch.zeros(6, 1, hidden_size).to(device)
    input_length = video_frames.size(1)
    with torch.no_grad():
        #for i in range(batch_size-1):
        #for n, index in enumerate(test_id):
            #batch_index = n % batch_size
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder.forward(
                    (video_frames[0,ei,:].unsqueeze(0)).unsqueeze(0), encoder_hidden)

            decoder_hidden = encoder_hidden
            input_token=torch.ones(1,1).type(torch.LongTensor).to(device)
            captions=[]
            caption = ""

            for seq in range(max_len):

                #input_token = torch.ones(1,1).type(torch.LongTensor).to(device)
                #input_token = input_token.unsqueeze(0)

                with torch.no_grad():
                    decoder_output, decoder_hidden = decoder(input_token, decoder_hidden)
                decoder_output = decoder_output.argmax(dim=1)
                caption += voc.index2word[str(int(decoder_output))] + " "
                input_token = decoder_output.unsqueeze(0)

            captions.append(caption)
            captions = captions[0].split()
            generated_captions=[]

            for i,k in enumerate(captions):
                if k=='eos':
                    continue
                elif k =='pad':
                    continue
                else:
                    generated_captions.append(k)

            string = ' '.join(generated_captions[:])
            #print(f'predicted caption: {string}')
            #print(generated_captions)
    prediction.write(string + "\n")
    #generated_caption.write(caption + "\n")
    return generated_captions
def test():
    print_test_loss_total = 0  # Reset every print_every
    plot_test_loss_total = 0  # Reset every plot_every
    criterion = nn.NLLLoss()
    video_frames = torch.zeros(1, 8, 4032).to(device)
    #target_tensor = torch.zeros(batch_size, train_tokens.shape[1]).to(device)
    trgs = []
    pred_trgs = []
    reference=[]
    print_bleu_1_total = 0
    print_bleu_2_total = 0
    print_bleu_3_total = 0
    print_bleu_4_total = 0
    #for iters in tqdm(range(1, n_iters + 1)):
    import csv
    #with open('results/nasnet_epoch_blue_scores.csv', mode='w') as new_file:
    #    new_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #    new_writer.writerow(['Iteration', 'BLEU-1', 'BLEU-2','BLEU-3','BLEU-4'])
    for iters in tqdm(range(1, n_iters + 1)):
        encoder = torch.load('model_nasnet_6_layer/%s_epoch_encoder.pth' % (iters))
        decoder = torch.load('model_nasnet_6_layer/%s_epoch_decoder.pth' % (iters))
        prediction = open("predict_nasnet_6_layer/prediction_%s.txt"% (iters), "w+")
        #encoder = torch.load('model_incep_3_layer/15_epoch_encoder.pth')
        #decoder = torch.load('model_incep_3_layer/15_epoch_decoder.pth')
    #encoder = torch.load('best_encoder.pth' )
    #decoder = torch.load('best_decoder.pth' )

        encoder.train()
        decoder.train()

        encoder.eval()
        decoder.eval()
        #import csv
    #    with open('results/nasnet_blue_scores.csv', mode='w') as new_file:
    #        new_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #        new_writer.writerow(['Iteration', 'BLEU-1', 'BLEU-2','BLEU-3','BLEU-4'])
        count=0

        for n, index in enumerate(test_id):

            if n==0:
                reference.append(referance_caption[n])
            elif index == test_id[n-1]:
                reference.append(referance_caption[n])
            else:
                reference=[]
                reference.append(referance_caption[n])

            if n==len(test_id)-1:
                #print(index)
                #print(reference)
                video_frames[0] = torch.load('nasnet_feature_new/' + index + '.pt')  # encoder input
                pred_test = generate_caption(encoder=encoder, decoder=decoder, video_frames=video_frames,prediction=prediction)
                count += 1
                weights = (1, 0, 0, 0)
                bleu_1 = sentence_bleu(reference, pred_test, weights)
                weights = (0.5, 0.5, 0, 0)
                bleu_2 = sentence_bleu(reference, pred_test, weights)
                weights = (0.3, 0.3, 0.3, 0)
                bleu_3 = sentence_bleu(reference, pred_test, weights)
                weights = (0.25, 0.25, 0.25, 0.25)
                bleu_4 = sentence_bleu(reference, pred_test, weights)
                #print(f"iteration:{count}")
                #print(f"BLEU-1: {bleu_1}")
                #print(f"BLEU-2: {bleu_2}")
                #print(f"BLEU-3: {bleu_3}")
                #print(f"BLEU-4: {bleu_4}")
                print_bleu_1_total += bleu_1
                print_bleu_2_total += bleu_2
                print_bleu_3_total += bleu_3
                print_bleu_4_total += bleu_4
                print_bleu_1_avg = print_bleu_1_total / count
                print_bleu_2_avg = print_bleu_2_total / count
                print_bleu_3_avg = print_bleu_3_total / count
                print_bleu_4_avg = print_bleu_4_total / count
                #print(f"iteration:{count}")
                print(f"BLEU-1: {print_bleu_1_avg}")
                print(f"BLEU-2: {print_bleu_2_avg}")
                print(f"BLEU-3: {print_bleu_3_avg}")
                print(f"BLEU-4: {print_bleu_4_avg}")
                print_bleu_1_total=0
                print_bleu_2_total = 0
                print_bleu_3_total = 0
                print_bleu_4_total = 0
                #new_writer.writerow((iters, print_bleu_1_avg, print_bleu_2_avg, print_bleu_3_avg, print_bleu_4_avg))
            elif index != test_id[n+1]:
                #print(index)
                #print(reference)
                video_frames[0] = torch.load('nasnet_feature_new/' + index + '.pt')  # encoder input
                pred_test = generate_caption(encoder=encoder, decoder=decoder, video_frames=video_frames,prediction=prediction)
                count += 1
                weights = (1, 0, 0, 0)
                bleu_1 = sentence_bleu(reference, pred_test, weights)
                weights = (0.5, 0.5, 0, 0)
                bleu_2 = sentence_bleu(reference, pred_test, weights)
                weights = (0.3, 0.3, 0.3, 0)
                bleu_3 = sentence_bleu(reference, pred_test, weights)
                weights = (0.25, 0.25, 0.25, 0.25)
                bleu_4 = sentence_bleu(reference, pred_test, weights)
                #print(f"iteration:{count}")
                #print(f"BLEU-1: {bleu_1}")
                #print(f"BLEU-2: {bleu_2}")
                #print(f"BLEU-3: {bleu_3}")
                #print(f"BLEU-4: {bleu_4}")
                #new_writer.writerow((count, bleu_1, bleu_2, bleu_3, bleu_4))
                print_bleu_1_total += bleu_1
                print_bleu_2_total += bleu_2
                print_bleu_3_total += bleu_3
                print_bleu_4_total += bleu_4



            else:
                continue




def samplevid(id):
    video_frames = torch.zeros(1, 8, 2048).to(device)
    # target_tensor = torch.zeros(batch_size, train_tokens.shape[1]).to(device)
    for iters in tqdm(range(1, n_iters + 1)):
        encoder = torch.load('model_incep_3_layer/%s_epoch_encoder.pth' % (iters))
        decoder = torch.load('model_incep_3_layer/%s_epoch_decoder.pth' % (iters))
        encoder.train()
        decoder.train()
        trgs = []
        pred_trgs = []
        encoder.eval()
        decoder.eval()

    print(f'id:{id}')
    video_frames[0] = torch.load('features/' + id + '.pt')
    generate_caption(encoder=encoder, decoder=decoder, video_frames=video_frames)


n_iters=50
"""
for iters in tqdm(range(1, n_iters + 1)):
    encoder=torch.load('model/%s_epoch_encoder.pth'% (iters))
    decoder=torch.load('model/%s_epoch_decoder.pth'% (iters))
"""
#bleu_1,bleu_2,bleu_3,bleu_4=test()
test()
#print(f"BLEU-1: {bleu_1}")
#print(f"BLEU-2: {bleu_2}")
#print(f"BLEU-3: {bleu_3}")
#print(f"BLEU-4: {bleu_4}")
"""
########################## Sample Video ###############################
#id='BnJUWwSx1kE_11_22'
#id='N6SglZopfmk_97_111'
#id='YmXCfQm0_CA_109_120'
#id='8yS2wqwActs_2_14'
#id='SzEbtbNSg04_71_93'
#id='QHkvBU8diwU_1_18'
#id='QMJY29QMewQ_42_52'

samplevid(id)

import cv2
video = cv2.VideoCapture('YouTubeClips/' + id + '.avi')
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, view = video.read()
    cv2.imshow(id,view)

    if cv2.waitKey(25) & 0xFF == ord('t'):
        break

video.release()
cv2.destroyAllWindows()
"""