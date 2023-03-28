import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import torchvision.transforms as transforms

from dataCollector import getLinkDictFromCSV

def convertLabelToVec(labels, num_classes):
    one_hot_labels = []
    class_labels = []
    for label in labels:
        if label not in class_labels:
            class_labels.append(label)
        v = [0] * num_classes
        ind = class_labels.index(label)
        v[ind] = 1
        one_hot_labels.append(v)
    return one_hot_labels

def loadDataFiles(num_classes=45, path=""):
    images_files = []
    labels = []
    if path == "":
        datapath = os.path.join(os.getcwd(), "data")
    else:
        datapath = path
    for podcast in os.listdir(datapath):
        if num_classes == 0:
            break
        print(podcast)
        podcastPath = os.path.join(datapath, podcast)
        for episode in os.listdir(podcastPath):
            episodePath = os.path.join(podcastPath, episode)
            for imageFile in os.listdir(episodePath):
                if imageFile.endswith('.png'):
                    img_path = os.path.join(episodePath, imageFile)
                    images_files.append(img_path)
                    labels.append(podcast)
        num_classes = num_classes - 1
    return images_files, labels

def getPixelValues(image_files):
    x_data = []
    tensor_transform = transforms.Compose([transforms.ToTensor()])

    for img_path in image_files:
        if len(x_data) > 42000:
            break
        img = Image.open(img_path).convert('RGB')
        img_tensor = tensor_transform(img).to(torch.float32)
        img_mean, img_std = img_tensor.mean([1,2]), img_tensor.std([1,2])
        zero_std_indices = (img_std == 0)
        img_std[zero_std_indices] = 0.000001
        normalize_transform = transforms.Compose([transforms.Normalize(img_mean, img_std)])
        normalized_img_tensor = normalize_transform(img_tensor)
        if normalized_img_tensor.shape[0] != 3 or normalized_img_tensor.shape[1] != 240 or normalized_img_tensor.shape[2] != 426:
            print(img_path)
            os.remove(img_path)
        x_data.append(normalized_img_tensor)
    
    return x_data

def getImageDataVectors(image_files, labels, batch_indices, num_classes):
    batch_labels = []
    batch_image_files = []
    for i in batch_indices:
        batch_labels.append(labels[i])
        batch_image_files.append(image_files[i])

    y_data = convertLabelToVec(batch_labels, num_classes)
    x_data = getPixelValues(batch_image_files)

    x_data = torch.stack(x_data)
    y_data = torch.tensor(y_data).to(torch.float32)
    return x_data, y_data


    
