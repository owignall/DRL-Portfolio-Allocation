from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def print_sents(text):
    print(text)
    print("")
    print("HuggingFace")
    classifier = pipeline("sentiment-analysis")
    print(classifier([text]))

    print("\nTextBlob")
    print(TextBlob(text).sentiment)

    print("\nVader")
    analyzer = SentimentIntensityAnalyzer()
    print(analyzer.polarity_scores(text))


print_sents("Amazon posts biggest profit ever at height of pandemic in U.S.")