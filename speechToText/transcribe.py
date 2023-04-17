import argparse
import os
from speechToText import utils
from speechToText import mp4_to_mp3
import moviepy
import moviepy.editor


#"5086f7f974ef468cb4c631b7b188f8ac"
def speechToText(apiKey, mp4File):
    class args:
        def __init__(self, api_key, local, audio_file):
            self.api_key = api_key
            self.local = local
            self.audio_file = audio_file

    if ("mp4" in mp4File):
        converter = mp4_to_mp3.mp4_to_mp3(mp4File, "")
        converter.convert()
        args.audio_file = converter.mp3_name
    else:
        args.audio_file = mp4File

    
    
    args.api_key = apiKey
    args.local = True
    args.audio_file = mp4File
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('audio_file', help='url to file or local audio filename')
    # parser.add_argument('--local', action='store_true', help='must be set if audio_file is a local filename')
    # parser.add_argument('--api_key', action='store', help='<YOUR-API-KEY>')

    # args = parser.parse_args()

    if args.api_key is None:
        args.api_key = os.getenv("AAI_API_KEY")
        if args.api_key is None:
            raise RuntimeError("AAI_API_KEY environment variable not set. Try setting it now, or passing in your "
                               "API key as a command line argument with `--api_key`.")

    # Create header with authorization along with content-type
    header = {
        'authorization': args.api_key,
        'content-type': 'application/json'
    }

    if args.local:
        # Upload the audio file to AssemblyAI
        upload_url = utils.upload_file(args.audio_file, header)
    else:
        upload_url = {'upload_url': args.audio_file}

    # Request a transcription
    transcript_response = utils.request_transcript(upload_url, header)

    # Create a polling endpoint that will let us check when the transcription is complete
    polling_endpoint = utils.make_polling_endpoint(transcript_response)

    # Wait until the transcription is complete
    utils.wait_for_completion(polling_endpoint, header)

    # Request the paragraphs of the transcript
    paragraphs = utils.get_paragraphs(polling_endpoint, header)

    # Save and print transcript
    with open('transcript.txt', 'w') as f:
        for para in paragraphs:
            #print(para['text'] + '\n')
            f.write(para['text'] + '\n')
    
if __name__ == '__main__':
    speechToText("5086f7f974ef468cb4c631b7b188f8ac", r"C:\Users\james\Documents\Sound Recordings\recording (8).m4a")

