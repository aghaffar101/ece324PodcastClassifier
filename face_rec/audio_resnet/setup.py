from fastai.vision.all import *
from fastaudio.core.all import *

# ensure the dataset is downloaded to the local machine
path = untar_data(URLs.ESC50)

# ensure all the pretrained weights are downloaded
models = [resnet18]
for model in models:
    model(pretrained=True)