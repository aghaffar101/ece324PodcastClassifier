from youtubesearchpython import VideosSearch
import json
import webbrowser
import re

def youtubeSearch(mainquery):

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    mainquery = emoji_pattern.sub(r'', mainquery)
    videosSearch = VideosSearch(mainquery, limit = 30)
    youtubeResults = videosSearch.result()
    allResults = youtubeResults['result']

    with open('search/test.json', 'w') as fp:
            json.dump(allResults, fp, indent=4)

    return allResults





    