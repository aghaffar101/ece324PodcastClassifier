from asyncio import transports
from pydoc import describe
from threading import main_thread
from flask import Flask, request, render_template, jsonify
import json
import webbrowser
from pytube import YouTube
from Transcripts import ytranscripts
from Transcripts import TextCompare
from Transcripts import verificationTranscript
from search import youtubeSearch
from scraping import scrape
from speechToText import transcribe
from face_rec import main_video, simple_facerec
from yake_keywords import yake_keywords
import classifier


import os
import shutil, sys 



# Search instagram clips
def instaGram_search(text):
    path = "instagram_videos"
    dir_list = os.listdir(path)
    result_json = {}

    scrape.media(text)

    # get the download image by looking through the folder
    if "?igshid" in text:
        short_link = text.replace('https://www.instagram.com/reel/','')
        short_link = short_link.split("/?")
        short_link.pop()
        short_link = short_link[0]
    else:
        short_link = text.replace('https://www.instagram.com/reel/','').replace('/?utm_source=ig_web_copy_link','')


    
    directory = os.fsencode(f'instagram_videos/{short_link}')
    
    # Get the filenames for image and videos    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp4"):
            print("The Path is ")
            print(filename)
            break
    # des_filename = filename.replace("mp4", "txt")
    image_filename = filename.replace("mp4", "jpg")
    des_filename = filename.replace("mp4", "txt")
    

    # Get the transcipt from instagram video
    transcribe.speechToText("5086f7f974ef468cb4c631b7b188f8ac", f"instagram_videos/{short_link}/{filename}")

    podcast = classifier.classify_video(f"instagram_videos/{short_link}/{image_filename}", audiopath = "./new_audio.mp3", use_audio = False)
    print(podcast)

    #keyWordQuerySearch

    mainSearchQuery = podcast

    # Get transcript and instagram description from textfile to a string variable
    with open(f'transcript.txt', 'r', encoding="utf") as file:
        transcript_str = file.read().replace('\n', '')

    with open(f'instagram_videos/{short_link}/{des_filename}', 'r', encoding="utf") as file:
        des_str = file.read().replace('\n', '')

    
    #podcast = 

    #Facial Recognition of thumbnail initially
    names = main_video.findVideo(f"instagram_videos/{short_link}/{filename}")
    yakeKeywordsQuery = yake_keywords.yakeKeywords(f'transcript.txt')


    for name in names:
        mainSearchQuery += name + " "
    mainSearchQuery += (yakeKeywordsQuery + " full interview podcast")
    result = youtubeSearch.youtubeSearch(mainSearchQuery)
    match = verificationTranscript.verifyYoutube(result)

    print("Second")
    webbrowser.open(f"https://www.youtube.com/results?search_query={mainSearchQuery}")
        

    # Delete the folder after every iteration
    shutil.rmtree(f"instagram_videos/{short_link}")



    return match, names
    

if __name__ == '__main__':
    result = instaGram_search("https://www.instagram.com/reel/CnrVHmQJf_l/?utm_source=ig_web_copy_link")
    with open('test23.json', 'w') as fp:
            json.dump(result, fp, indent=4)