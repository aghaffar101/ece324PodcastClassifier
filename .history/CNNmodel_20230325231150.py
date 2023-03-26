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
    def __init__(self, height, width, channels, kernelSize=3, numClasses=10):
        super(CNNClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernelSize), nn.ReLU(),
            nn.Conv2d(32, 64, kernelSize), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernelSize), nn.ReLU(),
            nn.Conv2d(128, 128, kernelSize), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernelSize), nn.ReLU(),
            nn.Conv2d(256, 256, kernelSize), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        fc1_input_dim = self.compute_fc1_input_dim(height, width, kernelSize)
        #print("fc1 input dims, ", fc1_input_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(fc1_input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, numClasses)
        )

    def compute_fc1_input_dim(self, height, width, kernelSize):
        dimReduction = kernelSize - 1
        
        for _ in range(3):
            height = (height - 2 * dimReduction) // 2
            width = (width - 2 * dimReduction) // 2
        
        return 256 * height * width


    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
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
    
    model_save_path = "model_weights/"
    os.makedirs(model_save_path, exist_ok=True)

    num_epochs = 5
    epochsLis = np.arange(num_epochs)
    trainLossLis = np.empty(shape=(num_epochs))
    testLossLis = np.empty(shape=(num_epochs))


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

    #print("x_data", x_data)
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
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    device = torch.device("cpu")
    model.to(device)
    
    num_epochs = 5

    import gzip 
    import pickle
    """
    model_save_path = "model_weights/"
    os.makedirs(model_save_path, exist_ok=True)

    epochsLis = np.arange(num_epochs)
    trainLossLis = np.empty(shape=(num_epochs))
    testLossLis = np.empty(shape=(num_epochs))

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, device,)
        test_loss = test(model, test_dataloader, device,)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss}")
        trainLossLis[epoch] = train_loss
        testLossLis[epoch] = test_loss

        # Save model weights (for the last epoch only)
        if epoch == num_epochs-1:
            with gzip.open(os.path.join(model_save_path, f"model_epoch_{epoch+1}.pkl.gz"), 'wb') as f:
                pickle.dump(model.state_dict(), f)
        """
    
    import matplotlib.pyplot as plt

    #plt.plot(epochsLis, trainLossLis, testLossLis)
    #plt.xlabel("num epochs")
    #plt.ylabel("loss")
    #plt.show()


    # CODE TO LOAD THE TRAINED MODEL 
    loaded_model = CNNClassifier(height, width, channels, numClasses=len(y_data[0]))

    # load the saved weights - from a specific epoch # (in this case it is epoch 5 = num_epochs)
    with gzip.open(f"model_weights/model_epoch_{num_epochs}.pkl.gz", 'rb') as f:
        loaded_weights = pickle.load(f)

    loaded_model.load_state_dict(loaded_weights)
    loaded_model.to(device)

    # verifying that the loaded model is correct 
    test_loss = test(loaded_model, test_dataloader, device,)
    print(f"Epoch: {num_epochs}, Train Loss: {num_epochs:.4f}, Test Loss: {test_loss}")