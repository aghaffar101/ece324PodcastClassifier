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


import os
import shutil, sys 



# Initial route for flask app
# app = Flask(__name__)
# @app.route('/')
# def main():

#     return render_template('index.html')



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


    #keyWordQuerySearch

    mainSearchQuery = ""

    # Get transcript and instagram description from textfile to a string variable
    with open(f'transcript.txt', 'r', encoding="utf") as file:
        transcript_str = file.read().replace('\n', '')

    with open(f'instagram_videos/{short_link}/{des_filename}', 'r', encoding="utf") as file:
        des_str = file.read().replace('\n', '')

    
    

    #Facial Recognition of thumbnail initially
    names = main_video.findVideo(f"instagram_videos/{short_link}/{filename}")
    yakeKeywordsQuery = yake_keywords.yakeKeywords(f'transcript.txt')




    if (names):
        print("Person Found")

        for name in names:
            mainSearchQuery += name + " "
        mainSearchQuery += (yakeKeywordsQuery + " full interview podcast")
        result = youtubeSearch.youtubeSearch(mainSearchQuery)
        match = verificationTranscript.verifyYoutube(result)

        print("Second")
        webbrowser.open(f"https://www.youtube.com/results?search_query={mainSearchQuery}")
        
        
    else:

        #Description
        des_str = des_str.replace("#", " ")
        print("The description: ", des_str)
        mainSearchQuery += des_str + yakeKeywordsQuery
        print(mainSearchQuery)
        result = youtubeSearch.youtubeSearch(mainSearchQuery)
        match = verificationTranscript.verifyYoutube(result)
        webbrowser.open(f"https://www.youtube.com/results?search_query={mainSearchQuery}")
        names = "No one was found"
        

    # Delete the folder after every iteration
    shutil.rmtree(f"instagram_videos/{short_link}")



    return match, names


# Short video
def shortsSearch(text):
    yt = YouTube(text)
   
    #Download Scraped Video
    video = yt.streams.get_highest_resolution()
    video.download()
    print("Yayyy!! Download Completed!!!")

    #To Download Audio File
    audio = yt.streams.filter(only_audio=True).first()
    audio.download()

    out_file = video.download(output_path=".")
  
    # save the file
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    
    
    
    transcribe.speechToText("5086f7f974ef468cb4c631b7b188f8ac", new_file)


    result = youtubeSearch.youtubeSearch(yt.title)
    match = verificationTranscript.verifyYoutube(result)

    
    webbrowser.open(f"https://www.youtube.com/results?search_query={yt.title}")

    os.remove(new_file)

    return jsonify(match)




# @app.route('/', methods=['POST'])
# def my_form_post():

#     #Get Raw data from request
#     text = request.form['text']

#     if ("instagram" in text):
#         result = instaGram_search(text)
#         with open('test23.json', 'w') as fp:
#             json.dump(result, fp, indent=4)
#         return result
    
#     elif ("youtube" in text):
#         return shortsSearch(text)
#     else:
#         pass
    

if __name__ == '__main__':
    result = instaGram_search("https://www.instagram.com/reel/CnrVHmQJf_l/?utm_source=ig_web_copy_link")
    with open('test23.json', 'w') as fp:
            json.dump(result, fp, indent=4)