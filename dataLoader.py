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

def adjustDirectory(path=""):
    # we want to use the image folder type in pytorch, 
    # where each class is a subdir of 'data', then its pictures
    if path == "":
        datapath = os.path.join(os.getcwd(), "data")
    else:
        datapath = path
    for podcast in os.listdir(datapath):
        if (podcast == 'train') or (podcast == 'val'):
            continue
        print(podcast)
        podcastPath = os.path.join(datapath, podcast)
        count = 0
        for episode in os.listdir(podcastPath):
            episodePath = os.path.join(podcastPath, episode)
            for imageFile in os.listdir(episodePath):
                if imageFile.endswith('.png'):
                    img_path = os.path.join(episodePath, imageFile)
                    new_img = str(count) + '.png'
                    new_img_path = os.path.join(podcastPath, new_img)
                    count+=1
                    while True:
                        try:
                            os.rename(img_path, new_img_path)
                            break
                        except:
                            new_img = str(count) + '.png'
                            new_img_path = os.path.join(podcastPath, new_img)
                            count += 1
                    
    return

def getImageCounts(path=""):
    counts = {}
    for podcast in os.listdir(path):
        if podcast == "train" or podcast == "val" or podcast == "test":
            continue
        print(podcast)
        podcastPath = os.path.join(path, podcast)
        count = 0
        for imageFile in os.listdir(podcastPath):
            if imageFile.endswith('.png'):
                count += 1
        counts[podcast] = count
    return counts

def splitToTrainValid(train_ratio=0.8, path=""):
    if path == "":
        datapath = os.path.join(os.getcwd(), "data")
    else:
        datapath = path
    train_path = os.path.join(datapath, "train")
    valid_path = os.path.join(datapath, "val")
    # first we need a count of how much data there is
    counts = getImageCounts(datapath)
    for podcast in counts.keys():

        podcast_train_path = os.path.join(train_path, podcast)
        if not os.path.exists(podcast_train_path):
            os.mkdir(podcast_train_path)
        podcast_valid_path = os.path.join(valid_path, podcast)
        if not os.path.exists(podcast_valid_path):
            os.mkdir(podcast_valid_path)

        num_images = counts[podcast]
        indices = np.arange(num_images)
        np.random.shuffle(indices)
        num_train = int(num_images * train_ratio)
        train_indices = indices[:num_train]
        valid_indices = indices[num_train:]
        podcastPath = os.path.join(datapath, podcast)
        count = 0
        for imageFile in os.listdir(podcastPath):
            if imageFile.endswith('.png'):
                img_path = os.path.join(podcastPath, imageFile)
                if count in train_indices:
                    new_img_path = os.path.join(podcast_train_path, imageFile)
                else:
                    new_img_path = os.path.join(podcast_valid_path, imageFile)
                os.rename(img_path, new_img_path)
                count += 1

    return


if __name__ == "__main__":
    #adjustDirectory("alldata")
    splitToTrainValid(path="alldata")

    
