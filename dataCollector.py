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
        print("Could not find stream for:", video.title)
        return num_added
    
    new_path = os.path.join(path, video.title)

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
    # get a random sample

    # we want it to underestimate

    num_frames = (video.length * vid_to_download.fps) -  (vid_to_download.fps * 5)

    frame_indices = np.arange(num_frames)
    frames_to_pick = np.random.permutation(frame_indices)[:num_images]

    while True:
        notDone, frame = cap.read()
        if not notDone:
            cap.release()
            break
        if chooseRandomly:
            if i in frames_to_pick:
                if os.path.exists(f"{new_path}/{i}.png"):
                    index_to_change = np.where(frames_to_pick==i)[0][0]
                    frames_to_pick[index_to_change] += 1
                    i += 1
                    continue
                cv2.imwrite(f"{new_path}/{i}.png", frame)
                num_added += 1
        # if we arent choosing randomly, just choose the smallest index
        else:
            if num_added < num_images:
                if os.path.exists(f"{new_path}/{i}.png"):
                    i += 1
                    continue
                cv2.imwrite(f"{new_path}/{i}.png", frame)
                num_added += 1
            else:
                cap.release()
                break
        i += 1
    
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

    one_video_num_images = num_images / len(pl.videos)

    if one_video_num_images < 1:
        stop_early = True

    count = 0

    for video in pl.videos:
        if stop_early:
            if count == num_images:
                break
            images_added = downloadFramesOneVideo(video, 1, new_path, False)
        else:
            images_added = downloadFramesOneVideo(video, one_video_num_images, new_path, chooseRandomly)
        count += images_added
        print(images_added, "images added from video:", video.title)
    
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

if __name__ == "__main__":
    links_dict = getLinkDictFromCSV(csv_filename='playlistLinks.csv')
    downloadFramesToPath(links_dict=links_dict, num_images=200)
    print(path_weird_characters)