import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import os
from PIL import Image

from torch.utils.data import random_split
from dataLoader import getImageDataVectors


'''
### use multiple frames to finally make a prediction 

## for each podcast clip (20s long)
## do the naive approach: 

## Make a prediction for each frame/img from the podcast clip and then classify it as the podcast name 
## with the highest probability ***

## then for all the clips repeat *** and do naive bayes assuming that all the individual frames are independent

'''

class CNNClassifier(nn.Module):

    def __init__(self, numClasses=10):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)


        self.fc1 = nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, numClasses) 
        # final layer is the number of podcast titles number of nodes 


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        print(x.shape)
        
        print(x.shape[1] * x.shape[2] * x.shape[3])
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


x_data, y_data = getImageDataVectors()

print(x_data.shape, y_data.shape)

model = CNNClassifier(numClasses=len(y_data[0]))

output = model.forward(x_data)
print(output)