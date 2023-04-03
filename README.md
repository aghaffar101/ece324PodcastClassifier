# ece324PodcastClassifier
classify podcasts described in playlistLinks.csv using image classification, audio classification, facial and voice recognition.

Files:

If you want to see our podcasts:

playlistLinks.csv: csv of all our playlists

pl.csv: dummy csv when we only want data from a subset of playlistLinks.csv

If you want to see our data collection:

dataCollector.py: Where we download and extract frames from YouTube given the playlist links in playlistLinks.csv

dataLoader.py : Acts as a library which contains functions that convert string labels to one hot vectors, convert the image files in ur data directory into their RGB 
pixel values and also loads them

If you want to see our CNN model:

CNNModel.py: has code for declaring the CNN class, training, validation and testing. 
