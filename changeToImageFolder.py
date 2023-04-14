import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import torchvision.transforms as transforms

def adjustDirectory(path=""):
    # we want to use the image folder type in pytorch, 
    # where each class is a subdir of 'data', then its pictures
    if path == "":
        datapath = os.path.join(os.getcwd(), "data")
    else:
        datapath = path
    for podcast in os.listdir(datapath):
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
    splitToTrainValid(path="alldata")