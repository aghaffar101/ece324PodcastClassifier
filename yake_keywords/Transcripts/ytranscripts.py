from youtube_transcript_api import YouTubeTranscriptApi


def getYoutubeTranscript(link):
    newLink = link.rsplit('v=', 1)[1]
    outls = []
    fullText = ""
    #print('hi')

    text = YouTubeTranscriptApi.get_transcript(newLink)
    for i in text:
        outext = (i['text']) + '\n'
        fullText += outext

    with open('op.txt', 'w') as opf:
        opf.write(fullText)

#getYoutubeTranscript("https://www.youtube.com/watch?v=zWQxNoqKE6E&t=64s")

