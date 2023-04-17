from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from resnet import initialize_model
from PIL import Image

def test_model(datapath):
    device = torch.device('cpu')
    data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    model, input_size = initialize_model(model_name="resnet", num_classes=45, feature_extract=True, use_pretrained=True)
    model.load_state_dict(torch.load("allclassmodel.pt"))
    model = model.to(device)
    model.eval() # switch to testing mode
    imageTensors = []
    individual_acc = 0
    averaged_acc = 0
    podcastLabel = 0
    num_clips = 0
    num_pics = 0
    # now i need to load the data
    for podcast in os.listdir(datapath):
        running_ind_acc = 0
        running_avg_acc = 0
        if podcast == "Below The Belt":
            podcastLabel += 1
            continue
        labels = torch.ones(10) * podcastLabel
        podcastPath = os.path.join(datapath, podcast)
        for clip in os.listdir(podcastPath):
            num_clips += 1
            imageTensors = []
            clipPath = os.path.join(podcastPath, clip)
            for imageFile in os.listdir(clipPath):
                num_pics += 1
                imagePath = os.path.join(clipPath, imageFile)
                img = Image.open(imagePath).convert('RGB')
                img_tensor = data_transforms['val'](img).to(torch.float32)
                imageTensors.append(img_tensor)

            imageTensors = torch.stack(imageTensors)
            outputs = model(imageTensors)
            _, preds = torch.max(outputs, 1)
            # preds is the individual predictions
            running_ind_acc += torch.sum(preds == labels)
            averaged_output = torch.mean(outputs, dim=0)
            _, avg_pred = torch.max(averaged_output, 0)
            correct = (podcastLabel == avg_pred)
            running_avg_acc += correct
            individual_acc += running_ind_acc
            averaged_acc += running_avg_acc

        print("For podcast:", podcast)
        print("Accuracy for individual frames = ", running_ind_acc/500)
        print("Accuracy when we take the average = ", running_avg_acc/50)
        podcastLabel += 1

    averaged_acc = (averaged_acc / num_clips)*100
    individual_acc = (individual_acc / num_pics)*100

    print("Total Individual Accuracy:", individual_acc)
    print("Total Averaged Accuracy:", averaged_acc)

if __name__ == "__main__":
    test_model(datapath="clipData")
    

    

        



            

