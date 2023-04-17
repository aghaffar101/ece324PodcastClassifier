import moviepy
import moviepy.editor

class mp4_to_mp3:
    mp3_name : str
    mp4_name : str
    def __init__(self, mp4_name, mp3_name):
        self.mp4_name = mp4_name
        self.mp3_name = mp3_name

    def convert(self):
        video = moviepy.editor.VideoFileClip(self.mp4_name, verbose= False)    # Put your file path in here
        _mp3_name = "new_audio.mp3"
        # Convert video to audio
        audio = video.audio
        audio.write_audiofile(_mp3_name)
        self.mp3_name = _mp3_name


