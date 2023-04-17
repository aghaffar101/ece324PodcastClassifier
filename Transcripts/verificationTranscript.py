from Transcripts import ytranscripts
from Transcripts import TextCompare
def verifyYoutube(videos):

    result_json = {
        "main_video" : [],
        "more_results" : []
    }

    print("Start")
    for vid in videos:

        link = vid['link']
        try:
            ytranscripts.getYoutubeTranscript(link)
            
        except:
            with open('op.txt', 'w') as opf:
                opf.write("ERROR Error aEr")
            print("Transcript ERROR")
        

        textComp = TextCompare.textCompare(r"transcript.txt", r"op.txt")
        isMatch = textComp.compare()
        if (isMatch):
            print('MATCH FOUND')
           
            result_json["main_video"].append(vid)
        else:
            result_json["more_results"].append(vid)

    return result_json
        


        