import torch
from logReg import LogisticRegressionModel

if __name__ == "__main__":
    model = LogisticRegressionModel(90, 45)
    W = torch.eye(45, 45)
    Z = torch.zeros(45, 45)
    A = torch.cat((W, Z), dim=1)
    print(A.shape)
    model.linear.weight = torch.nn.Parameter(A)
    model.linear.bias = torch.nn.Parameter(torch.zeros(1))
    torch.save(model.state_dict(), "logregmodel.pt")