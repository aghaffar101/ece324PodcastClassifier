from pytube import Playlist
import os

link = "https://www.youtube.com/playlist?list=PLhhgdzSsFu8hgqiJByRo_f-SAiEiGnln6"
pl = Playlist(link)
vid = pl.videos[0]
vid_stream = vid.streams.filter(only_video=True, resolution='240p').first()
print(vid_stream.fps)
print(vid.length)

#os.mkdir("test")
#os.rmdir("test")

#os.remove("killme.txt")