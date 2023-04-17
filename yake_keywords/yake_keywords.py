
import yake
# kw_extractor = yake.KeywordExtractor()
# text = """You had to make a bad drug legal, the worst choice was alcohol. It's definitely the case. And I'm saying that as somewhat of a fan of alcohol. Me too. Yes, I like it.
# But it's a bad drug. And this is also starting to become archaic as we get into drugs that are like people are using psychedelics more, where people people are a little bit curious about drugs that make you think instead of drugs that make you not think. Right? Yeah, because alcohol is kind of an escape into well, it's not full unconsciousness, although it certainly can be an alcohol also makes people aggressive. It's the only drug we know that actually makes people aggressive.
# So you see a massive effect on crime rates because half the people who murder someone are drunk. Drunk.
# """
# language = "en"
# max_ngram_size = 3
# deduplication_threshold = 0.2
# numOfKeywords = 20
# custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
# keywords = custom_kw_extractor.extract_keywords(text)
# for kw in keywords:
#     print(kw)

def yakeKeywords(textfile):
    with open(textfile, 'r') as file:
        text = file.read().replace('\n', ' ')
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.2
    numOfKeywords = 6
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    keywords_query = ''
    for kw in keywords:
        keywords_query += ' ' + kw[0]
    return keywords_query

# print(yakeKeywords("""You had to make a bad drug legal, the worst choice was alcohol. It's definitely the case. And I'm saying that as somewhat of a fan of alcohol. Me too. Yes, I like it.
# But it's a bad drug. And this is also starting to become archaic as we get into drugs that are like people are using psychedelics more, where people people are a little bit curious about drugs that make you think instead of drugs that make you not think. Right? Yeah, because alcohol is kind of an escape into well, it's not full unconsciousness, although it certainly can be an alcohol also makes people aggressive. It's the only drug we know that actually makes people aggressive.
# So you see a massive effect on crime rates because half the people who murder someone are drunk. Drunk.
# """))