from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# print(analyzer.polarity_scores("Google has a fantastic quater"))

with open("test_titles.txt", "r") as file:
    text = file.read()
titles = text.split("\n")

t_count = 0
a_count = 0 

for t in titles:
    if "Apple" in t:
        c = analyzer.polarity_scores(t)['compound']
        if c < - 0.5 or c > 0.5:
            print(t)
            print(analyzer.polarity_scores(t)['compound'])


# print(100 * (a_count / t_count))