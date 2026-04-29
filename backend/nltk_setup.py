import nltk

def setup():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except:
        nltk.download("vader_lexicon")

if __name__ == "__main__":
    setup()
