import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
from resnet import initialize_model
from dataLoader import loadDataFiles
NUM_CLASSES = 45

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, inp_feats):
        output = self.linear(inp_feats)
        output = torch.softmax(output, dim=1)
        return output

def train_model(datapath):
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
    model.load_state_dict(torch.load("model_state_dicts/allclassmodel.pt"))
    model = model.to(device)

    imageTensors = []
    
    