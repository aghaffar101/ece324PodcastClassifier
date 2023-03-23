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


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.y[idx]
        return x_i, y_i



class CNNClassifier(nn.Module):
    def __init__(self, numClasses=10):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, numClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = CNNClassifier()

x_data, y_data = getImageDataVectors()

custom_dataset = MyDataset(x_data, y_data)

train_ratio = 0.8
total_length = len(custom_dataset)
train_length = int(train_ratio * total_length)
test_length = total_length - train_length

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_length, test_length], generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)



def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for i, data in enumerate():
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    numElems = i + 1
    return running_loss / numElems



def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
   
    return accuracy 




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
for epoch in range(num_epochs):

    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    test_accuracy = test(model, test_dataloader, device)

    print(f"Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")