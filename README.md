# ece324PodcastClassifier

classify podcasts on YouTube described in playlistLinks.csv using image classification, audio classification, facial and voice recognition as well as NLP for keyword extraction.

# Try our model: 

The file classifier.py contains a function 'classify_video' which takes in a video path and returns the most likely podcast! We also supply code to download a youtube video given its link in that file, just follow the current code in main to see. Note: video will need to be in the 45 predefined classes in playlistLinks.csv for it to work.

app.py contains the code for running a search algorithm to attempt to find source videos close to (or exactly equal to!) the video you input.

# Navigating the Repository

# Polished Code Files:

Podcasts Used:

playlistLinks.csv: csv of all our playlists

pl.csv: dummy csv when we only want data from a subset of playlistLinks.csv

labels.csv: the labels that our model outputs for a given video

Data Collection:

dataCollector.py: Where we download and extract videos, audio, frames and spectograms from YouTube given the playlist links in playlistLinks.csv

dataLoader.py : Acts as a library which contains functions that convert string labels to one hot vectors, convert the image files in ur data directory into their RGB 
pixel values and also loads them. It also contains code to change the format of directories into the 'ImageFolder' dataset type in pytorch, which makes training much easier, and allows you to split images into a training and test set.

spectogram.py: Converts an mp3 file into an equivalent spectogram image for our audio classifier.

Models and Training:

CNNModel.py: has code for declaring the CNN class, training, validation and testing. 

resnet.py: has code for declaring the ResNet-18 class, loading the pretrained model, training and testing (called val, but we didnt tune hyperparameters)

Testing and Output:

classifier.py: has code for running our final product, equipped with a function for downloading a youtube video given its link. Can classify with and without audio, we recommend without audio if video is longer than a few minutes.

app.py: Code for running search algorithm to determine source videos close to (or equal to) the video inputted.

modelTest.py: Runs a comparitive study on the performance of image classification with and without averaging on any new arbitrary dataset.

# Folders:

Folders contains code that are supplementary, not used in the final product, or saved data/figures.

Transcripts: Contains code for downloading and interpreting transcripts from YouTube videos. 

face_rec : Contains work we implemented on facial recognition. Works fairly well but is not very useful in the case of 240p images. Used more in app.py algorithm.

model_state_dicts: Contains state_dicts for pytorch models we trained. Names are explanatory of the model.

resultGraphs: Contains pdfs of matplotlib plots that were generated throughout the training of our models. Name are self-explanatory.

Scraping: Contains code for scraping mediums other than YouTube (inc. Instagram and TikTok).

search: Contains code for reverse video search, an algorithm used to attempt to find a YouTube video from keywords (which we generate)

speechToText: Contains code for converting audio files into text manuscripts.

yakeKeywords: Contains code for keyword extraction using NLP method YAKE
