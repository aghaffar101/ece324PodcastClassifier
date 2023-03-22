import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import torchvision.transforms as transforms

from dataCollector import getLinkDictFromCSV

def convertLabelToVec(labels):
    one_hot_labels = []
    dict = getLinkDictFromCSV('playlistLinks.csv')
    class_labels = list(dict.keys())
    for label in labels:
        v = [0] * len(class_labels)
        ind = class_labels.index(label)
        v[ind] = 1
        one_hot_labels.append(v)
    return one_hot_labels

def loadDataFiles():
    images_files = []
    labels = []
    datapath = "data"
    for podcast in os.listdir(datapath):
        podcastPath = os.path.join(datapath, podcast)
        for episode in os.listdir(podcastPath):
            episodePath = os.path.join(podcastPath, episode)
            for imageFile in os.listdir(episodePath):
                if imageFile.endswith('.png'):
                    img_path = os.path.join(episodePath, imageFile)
                    images_files.append(img_path)
                    labels.append(podcast)
    return images_files, labels

def getPixelValues(image_files):
    x_data = []
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = tensor_transform(img).to(torch.float32)
        img_mean, img_std = img_tensor.mean([1,2]), img_tensor.std([1,2])
        zero_std_indices = (img_std == 0)
        img_std[zero_std_indices] = 0.000001
        normalize_transform = transforms.Compose([transforms.Normalize(img_mean, img_std)])
        normalized_img_tensor = normalize_transform(img_tensor)
        x_data.append(normalized_img_tensor)
    
    return x_data

def getImageDataVectors():
    image_files, labels = loadDataFiles()
    y_data = convertLabelToVec(labels)
    x_data = getPixelValues(image_files)

    return x_data, y_data

    
