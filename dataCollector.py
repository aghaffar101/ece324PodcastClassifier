from pytube import YouTube, Playlist
import os
import numpy as np
import csv
import cv2
from pydub import AudioSegment
import spectogram
from moviepy.editor import *
import gc

path_fail_ind = 0
path_weird_characters = []

def downloadVideo(video_link, output_path=""):
    yt = YouTube(video_link)
    try:
        vid_to_download = yt.streams.filter(resolution='240p').first()
    except:
        print("Could not find video stream")
        return
    try:
        vid_path = vid_to_download.download(output_path=output_path)
    except:
        print("Cannot download video:", yt.title)
        return
    return vid_path
    
    

def getLinkDictFromCSV(csv_filename):
    links_dict = {}
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row == []:
                continue
            link = row[0]
            label = row[1]
            links_dict[label] = link
    return links_dict

def downloadFramesOneVideo(video, num_images, path, chooseRandomly):
    global path_fail_ind, path_weird_characters
    num_added = 0
    try:
        vid_to_download = video.streams.filter(only_video=True, resolution='240p').first()
    except:
        print("Could not find stream for:", video)
        return num_added
    
    try:
        title = video.title
    except:
        title = None

    new_path = os.path.join(path, title)

    try:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
    except:
        # couldnt make a path with that name
        path_weird_characters.append(video.title)
        new_path = os.path.join(path, str(path_fail_ind))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        path_fail_ind += 1

    try:
        vid_path = vid_to_download.download(output_path=new_path)
    except:
        print("Cannot download video:", video.title)
        return num_added
    
    cap = cv2.VideoCapture(vid_path)
    i = 0

    num_frames = cap.get(7) # gets number of frames in capture

    frame_indices = np.arange(num_frames)
    frames_to_pick = np.random.permutation(frame_indices)[:num_images]
    frames_to_pick = np.sort(frames_to_pick)

    for frameInd in frames_to_pick:
        cap.set(1, frameInd)
        notDone, frame = cap.read()
        if not notDone:
            cap.release()
            break
        while os.path.exists(f"{new_path}/{frameInd}.png") and notDone:
            frameInd += 1
            cap.set(1, frameInd)
            notDone, frame = cap.read()

        if not notDone:
            cap.release()
            break
        
        cv2.imwrite(f"{new_path}/{frameInd}.png", frame)
        num_added += 1

    cap.release()
    
    os.remove(vid_path)
    return num_added

def downloadFramesOnePlaylist(playlist_link,podcast, num_images, path, chooseRandomly):
    global path_fail_ind, path_weird_characters
    
    try:
        pl = Playlist(playlist_link)
    except:
        print("error in making playlist for link:", playlist_link)
        return 0
    
    new_path = os.path.join(path, podcast)

    try:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
    except:
        # couldnt make a directory with that name, so I guess change it
        path_weird_characters.append(podcast)
        new_path = os.path.join(path, str(path_fail_ind))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        path_fail_ind += 1

    stop_early = False

    length = min(len(pl.videos), 10)

    one_video_num_images = num_images // length

    if one_video_num_images < 1:
        stop_early = True

    count = 0

    for video in pl.videos:
        if stop_early:
            if count >= num_images:
                break
            images_added = downloadFramesOneVideo(video, 1, new_path, False)
        else:
            if count >= num_images:
                break
            images_added = downloadFramesOneVideo(video, one_video_num_images, new_path, chooseRandomly)
        count += images_added
        try:
            title = video.title
        except:
            title = None
        print(images_added, "images added from video:", title)
    
    return count

def downloadFramesToPath(links_dict, num_images, path="", chooseRandomly=True):
    directory = "data"
    path = os.path.join(path, directory)

    try:
        os.mkdir(path)
        print("Making data directory")
    except:
        pass

    one_playlist_num_images = num_images // len(links_dict.keys())

    if one_playlist_num_images < 1:
        print("Need to add atleast one image per playlist")
        return

    print("Trying to add", one_playlist_num_images, "images per playlist")

    for podcast in links_dict.keys():
        playlist_link = links_dict[podcast]
        num_images_added = downloadFramesOnePlaylist(playlist_link, podcast, one_playlist_num_images, path, chooseRandomly)
        print(num_images_added, "imaged added from playlist:",podcast)
    return


def get_audio_snippet(time_start, audio):

    time_start = time_start * 1000

    # Open an mp3 file

    # pydub does things in milliseconds
    twenty_seconds = 20 * 1000

    # song clip of 20 seconds
    _20_second_snippet = audio[time_start:time_start+twenty_seconds]

    try :
        os.remove("convert.wav")
    except:
        pass

    # save file
    _20_second_snippet.export("convert.wav", format="wav")

def get_audio_data(csv_filename, clips_per_class, path=""):
    links_dict = getLinkDictFromCSV(csv_filename)
    for podcast in links_dict.keys():
        playlist_link = links_dict[podcast]
        playlist = Playlist(playlist_link)
        vid_links = playlist.video_urls
        playlist_length = min(len(vid_links), 20)
        clips_per_vid = clips_per_class // playlist_length
        if not os.path.exists(os.path.join(path, podcast)):
            os.mkdir(os.path.join(path, podcast))
        class_path = os.path.join(path, podcast)
        for i in range(playlist_length):
            link = vid_links[i*(len(vid_links)//playlist_length)]
            j = i*(len(vid_links)//playlist_length)
            while True:
                try:
                    download_audio(link, clips_per_vid, class_path, j)
                    break
                except:
                    j+=1
                    print("Error downloading audio, trying again")
                    link = vid_links[j]
            gc.collect()



            

def download_audio(vid_link, num_clips, path, vid_num):

        yt = YouTube(str(vid_link))

        # # extract only audio
        video = yt.streams.filter(only_audio=True).first()

        try:
            os.remove("temp.mp3")
        except:
            pass
        try:
            os.remove("temp.mp4")
        except:
            pass

        # download the file
        out_file = video.download(filename="temp.mp4")

        # video_info = yt_dlp.YoutubeDL().extract_info(
        #     url = vid_link,download=False
        # )
        # filename = f"temp.mp3"
        # options={
        #     'format':'bestaudio/best',
        #     'keepvideo':False,
        #     'outtmpl':filename,
        # }

        # with yt_dlp.YoutubeDL(options) as ydl:
        #     ydl.download([video_info['webpage_url']])


        # save the file
        length = yt.length
        sample_rate = length // num_clips
        audio = AudioSegment.from_file(r"C:\Users\james\Documents\U of T\UofTy3s2\ece324\ece324PodcastClassifier\temp.mp4", format="mp4")

        for i in range(num_clips):
            get_audio_snippet(i*sample_rate, audio)
            spectogram.get_spectogram("convert.wav", os.path.join(path, str(vid_num)+"_"+str(i)+".png"))
            gc.collect()





if __name__ == "__main__":
    # links_dict = getLinkDictFromCSV(csv_filename='pl.csv')
    # downloadFramesToPath(links_dict=links_dict, num_images=34000, path="D:", chooseRandomly=False)
    # print(path_weird_characters)
    get_audio_data("playlists_finish.csv", 1000, "./audio_data")
    #audio = AudioSegment.from_file(r"C:\Users\james\Documents\U of T\UofTy3s2\ece324\ece324PodcastClassifier\temp", format="mp3")
