# ece324PodcastClassifier

classify podcasts described in playlistLinks.csv using image classification, audio classification, facial and voice recognition as well as NLP for keyword extraction.

Try our model: the file classifier.py contains a function 'classify_video' which takes in a video path and returns the most likely podcast it is! We also supply code to download a youtube video given its link, just follow the current code in main to see. Note: video will need to be in the 45 predefined classes (more on that below) for it to work.

# Navigating the Repository

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
