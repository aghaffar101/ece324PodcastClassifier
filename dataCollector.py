from pytube import YouTube, Playlist
import os
import numpy as np
import csv
import cv2

path_fail_ind = 0
path_weird_characters = []

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

def get_audio_snippet(mp3_file, time_start):

    # Open an mp3 file
    audio = AudioSegment.from_file("testing.mp3",
                                format="mp3")

    # pydub does things in milliseconds
    twenty_seconds = 20 * 1000

    # song clip of 10 seconds from starting
    _20_second_snippet = audio[:twenty_seconds]

    # save file
    _20_second_snippet.export("_20_seconds.mp3", format="mp3")

def download_audio(links_dict, num_clips, path="", chooseRandomly=True):
    for link in links_dict.keys():
        # link of the video to be downloaded
        # url input from user
        yt = YouTube(str(link))

         # extract only audio
        video = yt.streams.filter(only_audio=True).first()

        # check for destination to save file

        destination = path

        # download the file
        out_file = video.download(output_path=destination)

        # save the file
        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp3'
        os.rename(out_file, new_file)

if __name__ == "__main__":
    links_dict = getLinkDictFromCSV(csv_filename='pl.csv')
    downloadFramesToPath(links_dict=links_dict, num_images=34000, path="D:", chooseRandomly=False)
    print(path_weird_characters)