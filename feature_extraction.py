
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
        for j,image_path in enumerate(sorted(glob(f"Frames/train/{file_name}_Frame_*.jpg"))):

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


        torch.save(features, f'features/{file_name}.pt')


image_path = 'Frames\\train'
video_path="YouTubeClips"
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