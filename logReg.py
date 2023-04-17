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

    imageTensors, labels = loadDataFiles(num_classes=45, path="clipData")
    

    losses = []
    train_accs = []
    valid_accs = []
    epochs_for_plot = []
    # define hyperparameters
    epochs = 10
    lr = 0.001
    linRegModel = LogisticRegressionModel(in_dim=3, out_dim=1)

    loss_fcn = torch.nn.BCELoss()
    optimiser = torch.optim.Adam(linRegModel.parameters(), lr=lr)

    # Perform the training loop
    for epoch in range(epochs):
        optimiser.zero_grad()
        outputs = linRegModel.forward(imageTensors)
        #print(outputs)
        loss = loss_fcn(torch.squeeze(outputs), labels)
        loss.backward()
        optimiser.step()

        # calculate and print the train and valid accuracy and loss
        if (epoch % 2 == 0) or epoch == (epochs - 1):
            res = 0
            res += np.sum(torch.squeeze(outputs).round().detach().numpy() == labels.detach().numpy())
            acc = 100 * res/(labels.size(0))

            print("EPOCH:",epoch,"The training loss is:",loss.item())
            print("The training accuracy is:",acc)

            validres = np.sum(torch.squeeze(linRegModel.forward(imageTensors)).round().detach().numpy() == labels.detach().numpy())
            validacc = 100*validres / (labels.size(0))
            print("The validation accuracy is:",validacc)

            epochs_for_plot.append(epoch)
            losses.append(loss.item())
            train_accs.append(acc)
            valid_accs.append(validacc)
        
    torch.save(linRegModel.state_dict(), "logregmodel.pt")