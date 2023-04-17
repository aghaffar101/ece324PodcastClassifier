import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
from resnet import initialize_model
from dataCollector import getLinkDictFromCSV
from dataCollector import downloadVideo, download_audio
from PIL import Image
import cv2
from speechToText.mp4_to_mp3 import mp4_to_mp3
from spectogram import get_spectogram
from logReg import LogisticRegressionModel

DEFAULT_DEVICE = torch.device('cpu')
NUM_CLASSES = 45
DATA_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def prob_from_frames(model, framesPath, device=DEFAULT_DEVICE, batch_size=32):

    imageFiles = os.listdir(framesPath)
    imageTensors = []
    avg_outputs = torch.zeros(NUM_CLASSES)
    i = 0

    while i < len(imageFiles):

        imageFile = imageFiles[i]
        imagePath = os.path.join(framesPath, imageFile)
        img = Image.open(imagePath).convert('RGB')
        img_tensor = DATA_TRANSFORM(img).to(torch.float32)
        imageTensors.append(img_tensor)

        if (i != 0) and ((i % batch_size) == 0):
            # our RAM is full, we need to run through the model, add the outputs, and then reset
            imageTensors = torch.stack(imageTensors)
            cur_output = model(imageTensors)
            summed_output = torch.sum(cur_output, dim=0)
            avg_outputs = avg_outputs + summed_output
            imageTensors = []
        
        i += 1

    if len(imageTensors) > 0:
        # there are some remaining
        imageTensors = torch.stack(imageTensors)
        cur_output = model(imageTensors)
        summed_output = torch.sum(cur_output, dim=0)
        avg_outputs = avg_outputs + summed_output
    
    avg_outputs = avg_outputs / len(imageFiles)
    return avg_outputs

def prob_from_audio(model, spectogramPath, device=DEFAULT_DEVICE):
    img = Image.open(spectogramPath).convert('RGB')
    img_tensor = DATA_TRANSFORM(img).to(torch.float32)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    model_output = model(img_tensor)
    return model_output


def convert_to_frames(videopath, gap_per_frame, output_path):
    ''' gap per frame has units seconds'''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    try:
        cap = cv2.VideoCapture(videopath)
    except:
        print("Video path is invalid")
        return
    
    notDone = True
    frameInd = 10
    while True:
        # set the frame to this particular index
        # Note on this: video is 30fps, so each frame is 1/30 of a second
        cap.set(1, frameInd)
        notDone, frame = cap.read()
        if not notDone:
            # no more frames in the video
            cap.release()
            break
        cv2.imwrite(f"{output_path}/{frameInd}.png", frame)
        frameInd += (gap_per_frame*30)

def classify_video(videopath, audiopath, device=DEFAULT_DEVICE, use_audio=True, timegap_per_frame=5):
    # get dict describing labels
    labelDict = getLinkDictFromCSV("labels.csv")
    labels = list(labelDict.values())
    # make the image into frames and audio
    convert_to_frames(videopath, gap_per_frame=timegap_per_frame, output_path="test_frames")
    
    # get the probability outputs for the image classifier
    image_model, input_size = initialize_model(model_name="resnet", num_classes=NUM_CLASSES, feature_extract=True, use_pretrained=True)
    image_model = image_model.to(device)
    image_model.load_state_dict(torch.load("model_state_dicts/allclassmodel.pt", map_location=torch.device('cpu')))
    image_model.eval() # switch to testing mode
    frame_probs = prob_from_frames(model=image_model, framesPath="test_frames")

    # get the probability ouputs for the audio classifier
    if use_audio:
        audio_model, input_size = initialize_model(model_name="resnet", num_classes=45, feature_extract=True, use_pretrained=True)
        audio_model = audio_model.to(device)
        audio_model.load_state_dict(torch.load("model_state_dicts/audioclassmodel.pt"))
        audio_model.eval()
        audio_probs = prob_from_audio(model=audio_model, spectogramPath= audiopath)
        # now grab the logistic regression model
        logRegModel = LogisticRegressionModel(in_dim=90, out_dim=45)
        logRegModel.load_state_dict(torch.load("model_state_dicts/logregmodel.pt"))
        both_probs = torch.cat((frame_probs, audio_probs), dim=0)
        output_probs = logRegModel(both_probs)
        _, pred = torch.max(output_probs, 0)
        return labels[pred.item()]
    else:
        _, pred = torch.max(frame_probs, 0)
        return labels[pred.item()]

if __name__ == "__main__":
    vid_link = "https://www.youtube.com/watch?v=OYhGxfP37us&ab_channel=JREClips"
    vid_path = downloadVideo(video_link=vid_link, output_path="testvideo")
    audio_path = download_audio(vid_link=vid_link, num_clips=1, path="testvideo", vid_num=1)
    podcast = classify_video(videopath=vid_path, audiopath=audio_path, use_audio=True, timegap_per_frame = 30)
    print("Video classified as:", podcast)





    







