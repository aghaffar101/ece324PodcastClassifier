import json
from unittest import result
import webbrowser
import os
import requests
import urllib
import bs4

from bs4 import BeautifulSoup

from requests_html import HTML
from requests_html import HTMLSession


def get_source(url):
    """Return the source code for the provided URL. 

    Args: 
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html. 
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def parse_results(response):
    
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".VwiC3b"
    
    results = response.html.find(css_identifier_result)

    output = []
    
    for result in results:

        item = {
            'title': result.find(css_identifier_title, first=True).text,
            'link': result.find(css_identifier_link, first=True).attrs['href'],
            'text': result.find(css_identifier_text, first=True).text
        }
        
        output.append(item)
        
    return output

def reverseImageSearch(filePath):
    searchUrl = 'http://www.google.com/searchbyimage/upload'
    multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
    response = requests.post(searchUrl, files=multipart, allow_redirects=False)
    fetchUrl = response.headers['Location']
    webbrowser.open(fetchUrl)

    source =  get_source(fetchUrl)
    results = parse_results(source)

    with open('search/test2.json', 'w') as fp:
            json.dump(results, fp, indent=4)

   
    

    return results
            


# def list_file_name(path):
#     fileList = os.listdir(path)
#     return(fileList)

# def inputImages(path):
#     allFiles = list_file_name(path)
#     allurls = []
#     for name in allFiles:
#         upload = cloudinary.uploader.upload("frames/"+name)
#         url = upload['url']
#         searchResult = searchAPI(url)
#         if (searchResult != "nomatch"):
#             return searchResult
#     return "nofound"

# def inputOneImage(pathName):
#     upload = cloudinary.uploader.upload(pathName)
#     url = upload['url']
#     searchResult = searchAPI(url)
#     return searchResult



