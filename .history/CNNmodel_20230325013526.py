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

from torch.utils.data import Dataset, DataLoader


'''
### use multiple frames to finally make a prediction 

## for each podcast clip (20s long)
## do the naive approach: 

## Make a prediction for each frame/img from the podcast clip and then classify it as the podcast name 
## with the highest probability ***

## then for all the clips repeat *** and do naive bayes assuming that all the individual frames are independent

'''

class CNNClassifier(nn.Module):

    def __init__(self, height, width, channels, numClasses=10):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        fc1_input_dim = self.compute_fc1_input_dim(height, width)

        self.fc1 = nn.Linear(fc1_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, numClasses) 
        # final layer is the number of podcast titles number of nodes 

    def compute_fc1_input_dim(self, height, width):
        # First convolution and pooling
        height = (height - 4) // 2
        width = (width - 4) // 2
        
        # Second convolution and pooling
        height = (height - 4) // 2
        width = (width - 4) // 2

        return 16 * height * width

    def forward(self, x):
        # flow of data

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # No softmax activation
        return x



class CustomTensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.argmax(axis=1)  # Convert one-hot encoded labels to class indices

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.y[idx]
        return x_i, y_i




def train(model, dataloader, device,):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        # enumerate is used to get the next batch of data 

        inputs, labels = data
        #print("inputs", inputs.shape)
        #print("label", labels)
        inputs, labels = inputs.to(device), labels.to(device)

        # zero out the gradients that were previously attached to weights 
        # to prevent accumulated gradients 
        optimizer.zero_grad() 
        
        # forward pass
        outputs = model(inputs) 
                
        # Calculate loss using raw logits
        loss = F.cross_entropy(outputs, labels)

        # Calculate accuracy using class probabilities
        _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
        correct = (predicted == labels).sum().item()
        
        loss.backward(retain_graph=True)

        optimizer.step()
        running_loss += loss.item()

    numElems = i + 1
    return running_loss / numElems




def test(model, dataloader, device,):
    model.eval() # test mode 
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
                
        # Calculate loss using raw logits
        loss = F.cross_entropy(outputs, labels)

        # Calculate accuracy using class probabilities
        _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
        correct = (predicted == labels).sum().item()

        running_loss += loss.item()
    
    numElems = i + 1
    return running_loss / numElems

   


if __name__ == "__main__":


    x_data, y_data = getImageDataVectors()


    print(x_data.shape)

    print(x_data.shape, y_data.shape)

    height, width, channels = x_data.shape[2], x_data.shape[3], x_data.shape[1]
    model = CNNClassifier(height, width, channels, numClasses=len(y_data[0]))

    output = model.forward(x_data)
    print(output)


    height, width, channels = x_data.shape[2], x_data.shape[3], x_data.shape[1]
    model = CNNClassifier(height, width, channels, numClasses=len(y_data[0]))


    #########################

    ## Test-train split:
    train_ratio = 0.8
    total_length = len(x_data)
    train_length = int(train_ratio * total_length)
    test_length = total_length - train_length

    x_train = x_data[:train_length]
    y_train = y_data[:train_length]

    x_test = x_data[train_length:]
    y_test = y_data[train_length:]

    train_dataset = CustomTensorDataset(x_train, y_train)
    test_dataset = CustomTensorDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # setting the loss function and the training optimizer 
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



    device = torch.device("cpu")

    model.to(device)
    
    num_epochs = 15
    epochsLis = np.arange(num_epochs)
    trainLossLis = np.empty(shape=(num_epochs))
    testLossLis = np.empty(shape=(num_epochs))

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, device,)
        test_loss = test(model, test_dataloader, device,)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss}")
        trainLossLis[epoch] = train_loss
        testLossLis[epoch] = test_loss

    import matplotlib.pyplot as plt

    plt.plot(epochsLis, trainLossLis, testLossLis)
    plt.xlabel("num epochs")
    plt.ylabel("loss")
    plt.show()


