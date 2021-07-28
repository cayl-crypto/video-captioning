from models import *
import torch
from tokenization import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
""""
## ADD 'Frames' folder to file path before running code.

def Video_to_Frames():
    # Run the above function and store its results in a variable.
    video_path="C:\\Users\\pc\\PycharmProjects\\video-captioning\\sample_video"
    path_to_save = "C:\\Users\\pc\\PycharmProjects\\video-captioning\\sample\\"

    get_filepaths(video_path,path_to_save)


def main():
    Video_to_Frames()


if __name__ == "__main__":
    main()


import torch
from torchvision import datasets, transforms
from Inception import inception_v3
from tqdm import tqdm
import numpy as np
from utils import *
from glob import glob

def get_filepath(video_path,batch_size=16):
    file_names = []  # List which will store all of the full filepaths.

    for root, directories, files in os.walk(video_path):
        for filename in files:
            file = os.path.basename(filename).split('.', 1)[0]
            file_names.append(file)  # Add it to the list.
    file_names.sort()

    #print(file_names)
    feature_extraction(file_names,batch_size=batch_size)
    return file_names  # Self-explanatory.

# Iterate each image
def feature_extraction(file_names,batch_size=16):

    # Will contain the feature

    for file_name in tqdm(file_names):
        features = torch.zeros(8, 2048)
        for j,image_path in enumerate(sorted(glob(f"sample/{file_name}_Frame_*.jpg"))):

          # Read the file
          img =preprocess(gray_to_rgb(load_image(image_path)))
            # Reshape the image. PyTorch model reads 4-dimensional tensor
            # [batch_size, channels, width, height]

          img = img.to(device)

            # We only extract features, so we don't need gradient
          with torch.no_grad():
                # Extract the feature from the image
            feature = model(img.unsqueeze(0))

          features[j]=feature


        torch.save(features, f'sample_features/{file_name}.pt')


image_path = 'sample'
video_path="sample_video"
batch_size=64
# Transform the image, so it becomes readable with the model
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = inception_v3(pretrained=True)

model.to(device)
model.eval()
get_filepath(video_path,batch_size=batch_size)

"""
batch_size=64
hidden_size = 256
def generate_caption(encoder,decoder, video_frames, max_len=20):

    voc = Voc()

    voc.load_vocabulary()
    encoder_hidden = torch.zeros(1, 1, hidden_size).to(device)
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
            print(f'predicted caption: {string}')
            #print(generated_captions)
    #generated_caption.write(caption + "\n")
    return generated_captions


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
    video_frames[0] = torch.load('sample_features/' + id + '.pt')
    generate_caption(encoder=encoder, decoder=decoder, video_frames=video_frames)

n_iters=15




id='1'

samplevid(id)

import cv2
video = cv2.VideoCapture('sample_video/' + id + '.mp4')
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, view = video.read()
    cv2.imshow(id,view)

    if cv2.waitKey(25) & 0xFF == ord('t'):
        break

video.release()
cv2.destroyAllWindows()

















