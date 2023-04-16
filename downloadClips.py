from pytube import YouTube, Playlist
import os
import numpy as np
import csv
import cv2
from pydub import AudioSegment
from dataCollector import getLinkDictFromCSV

def downloadClips(linksDict, clipsPerPodcast, clipSize, imagesPerClip):
    directory = "clipData"

    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Making data directory")
    
    for podcast in linksDict.keys():
        pl_link = linksDict[podcast]
        try:
            pl = Playlist(pl_link)
        except:
            print("error in making playlist for link:", pl_link)
            return
        new_path = os.path.join(directory, podcast)

        if not os.path.exists(new_path):
            os.mkdir(new_path)
        
        first_video = pl.videos[0]

        try:
            title = first_video.title
        except:
            title = "title didnt work"

        i = 0
        while True:
            try:
                vid_to_download = first_video.streams.filter(only_video=True, resolution='240p').first()
                break
            except:
                print("Could not find stream for:", podcast)
                first_video = pl.videos[i+1]
                i += 1
        
        try:
            vid_path = vid_to_download.download(output_path=new_path)
            print("Downloading video:", title)
        except:
            print("Cannot download video:", podcast)
        
        for i in range(clipsPerPodcast):
            # start after 10 minutes
            # youtube videos are 30fps
            clip_path = os.path.join(new_path, str(i))

            if not os.path.exists(clip_path):
                os.mkdir(clip_path)

            start_frame = 600*30 + (300*i)
            end_frame = start_frame + (clipSize*30)
            frames_to_download = np.arange(start_frame, end_frame, clipSize*30/imagesPerClip)
            
            cap = cv2.VideoCapture(vid_path)

            for frameInd in frames_to_download:
                cap.set(1, frameInd)
                notDone, frame = cap.read()
                if not notDone:
                    cap.release()
                    break
                cv2.imwrite(f"{clip_path}/{frameInd}.png", frame)

            cap.release()

        os.remove(vid_path)




if __name__ == "__main__":
    linksDict = getLinkDictFromCSV("playlistLinks.csv")
    downloadClips(linksDict, 50, 20, 10)
