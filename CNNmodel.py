import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split

import numpy as np
import os
from PIL import Image

from torch.utils.data import random_split
from dataLoader import loadDataFiles, getImageDataVectors
from dataCollector import getLinkDictFromCSV

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
        x = torch.softmax(x, dim=1)
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



def train_op(model,x_data,y_data,device):
    model.train()
    running_loss = 0.0
    
    model_save_path = "model_weights/"
    os.makedirs(model_save_path, exist_ok=True)

    inputs, labels = x_data.to(device), y_data.to(device)

    # zero out the gradients that were previously attached to weights 
    # to prevent accumulated gradients 
    optimizer.zero_grad() 
        
    # forward pass
    outputs = model(inputs) 
                
    # Calculate loss using raw logits
    loss = F.cross_entropy(outputs, labels)
        
    loss.backward(retain_graph=True)

    optimizer.step()
    running_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    _, targets = torch.max(labels, 1)
    correct = (predicted == targets).sum().item()

    return running_loss, correct


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
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()

        running_loss += loss.item()
    
    numElems = i + 1
    return running_loss / numElems

   


if __name__ == "__main__":
    num_classes = 2
    podcastDict = getLinkDictFromCSV('playlistLinks.csv')
    image_files, labels = loadDataFiles(num_classes = 2)
    height = 240
    width = 426
    channels = 3
    model = CNNClassifier(height, width, channels, numClasses=num_classes)

    #########################


    ## train-valid-test split: 0.7-0.15-0.15

    x_train_files, x_testvalid_files, y_train_labels, y_testvalid_labels = train_test_split(image_files, labels, test_size=0.3, random_state=69)
    x_valid_files, x_test_files, y_valid_labels, y_test_labels = train_test_split(x_testvalid_files, y_testvalid_labels, test_size=0.5, random_state=69)
    
    # train_ratio = 0.7
    # total_length = len(image_files)
    # train_length = int(train_ratio * total_length)
    # test_length = (total_length - train_length) // 2

    # x_train_files = image_files[:train_length]
    # y_train_labels = labels[:train_length]

    # x_test_files = image_files[train_length:train_length + test_length]
    # y_test_labels = labels[train_length: train_length + test_length]

    # x_valid_files = image_files[train_length+test_length:]
    # y_valid_labels = labels[train_length+test_length:]

    # train_dataset = CustomTensorDataset(x_train, y_train)
    # test_dataset = CustomTensorDataset(x_test, y_test)

    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # setting the loss function and the training optimizer 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    device = torch.device('cpu')
    model.to(device)
    
    num_epochs = 5

    import gzip 
    import pickle
    
    model_save_path = "model_weights/"
    os.makedirs(model_save_path, exist_ok=True)

    epochsLis = np.arange(num_epochs)
    trainLossLis = np.empty(shape=(num_epochs))
    validLossLis = np.empty(shape=(num_epochs))

    train_accs = np.empty(shape=(num_epochs))
    valid_accs = np.empty(shape=(num_epochs))

    batch_size = 32
    num_iters = len(x_train_files) // batch_size

    for epoch in range(num_epochs):
        avg_train_loss = 0
        avg_valid_loss = 0
        train_acc = 0
        valid_acc = 0

        # run training

        train_indices = np.arange(len(x_train_files))
        np.random.shuffle(train_indices)

        for it in range(0, len(x_train_files), batch_size):
            print(it)
            batch_indices = train_indices[it:it+batch_size]
            x_batch, y_batch = getImageDataVectors(x_train_files, y_train_labels, batch_indices, num_classes)
            train_loss, acc = train_op(model, x_batch, y_batch, device)
            avg_train_loss += train_loss
            train_acc += acc
        
        avg_train_loss = avg_train_loss / num_iters
        train_acc = train_acc / len(x_train_files)

        trainLossLis[epoch] = avg_train_loss
        train_accs[epoch] = train_acc

        # run validation

        valid_indices = np.arange(len(x_valid_files))
        np.random.shuffle(valid_indices)

        for it in range(0, len(x_valid_files), batch_size):
            print(it)
            batch_indices = valid_indices[it:it+batch_size]
            x_batch, y_batch = getImageDataVectors(x_valid_files, y_valid_labels, batch_indices, num_classes)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
                
            # Calculate loss using raw logits
            running_loss = F.cross_entropy(outputs, y_batch).item()
            avg_valid_loss += running_loss
            _, predicted = torch.max(outputs, 1)
            _, targets = torch.max(y_batch, 1)
            correct = (predicted == targets).sum().item()
            valid_acc += correct

        avg_valid_loss = avg_valid_loss / len(x_valid_files)
        valid_acc = valid_acc / len(x_valid_files)

        validLossLis[epoch] = avg_valid_loss
        valid_accs[epoch] = valid_acc

        print(f"Epoch: {epoch} Training Loss: {avg_train_loss} Validation Loss: {avg_valid_loss}.Training Accuracy: {train_acc} Validation Accuracy: {valid_acc}")
        # Save model weights (for the last epoch only)
        with gzip.open(os.path.join(model_save_path, f"model_epoch_{epoch+1}_classes{num_classes}.pkl.gz"), 'wb') as f:
            pickle.dump(model.state_dict(), f)

    import matplotlib.pyplot as plt

    plt.title("Training vs Validation Loss")
    plt.plot(epochsLis, trainLossLis, label="Training")
    plt.plot(epochsLis, validLossLis, label="Validation")
    plt.xlabel("num epochs")
    plt.ylabel("loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training vs Validation Accuracy")
    plt.plot(epochsLis, train_accs, label="Training")
    plt.plot(epochsLis, valid_accs, label="Validation")
    plt.xlabel("num epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='best')
    plt.show()

    # Now test the model 
    
    test_acc = 0
    test_indices = np.arange(len(x_test_files))
    np.random.shuffle(test_indices)

    for it in range(0, len(x_test_files), batch_size):
        batch_indices = test_indices[it:it+batch_size]
        x_batch, y_batch = getImageDataVectors(x_test_files, y_test_labels, batch_indices, num_classes)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch) 
                
        _, predicted = torch.max(outputs, 1)
        _, targets = torch.max(y_batch, 1)
        correct = (predicted == targets).sum().item()
        test_acc += correct

    test_acc = test_acc / len(x_test_files)
    print(f"Test Accuracy: {test_acc}")


    # # CODE TO LOAD THE TRAINED MODEL 
    # loaded_model = CNNClassifier(height, width, channels, numClasses=len(y_data[0]))

    # # load the saved weights - from a specific epoch # (in this case it is epoch 5 = num_epochs)
    # with gzip.open(f"model_weights/model_epoch_{num_epochs}.pkl.gz", 'rb') as f:
    #     loaded_weights = pickle.load(f)

    # loaded_model.load_state_dict(loaded_weights)
    # loaded_model.to(device)

    # # verifying that the model is loaded correctly 
    # test_loss = test(loaded_model, test_dataloader, device,)
    # print(f"Epoch: {num_epochs}, Train Loss: {num_epochs:.4f}, Test Loss: {test_loss}")