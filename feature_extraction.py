from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from Inception import inception_v3

DATA_DIR = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\Frames\\train'
LOG_DIR = 'logs'
BATCH_SIZE = 6
WORKERS = 6
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
FREEZE = True

# check for cuda and set gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])

train_set = datasets.ImageFolder(DATA_DIR, transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                           shuffle=True, num_workers=4)
classes = train_set.classes

# inception v3
model = inception_v3(pretrained=True)
if FREEZE:
    for param in model.parameters():
        param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
torch.save(model, 'features.pt')

