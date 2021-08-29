from textblob import TextBlob



# Textblob just averages sentiment of words and phrases

with open("other/sentiment_analysis/test_titles.txt", "r") as file:
    text = file.read()

titles = text.split("\n")

results = []

# for t in titles:
#     sent = TextBlob(t).sentiment
#     print({"polarity": sent.polarity, "subjectivity": sent.subjectivity})
#     results.append([t, sent.polarity, sent.subjectivity])

# results.sort(key=lambda x:x[1])

# for i in range(100):
#     print(results[i])

# print(TextBlob("NLP is great").sentiment)
# print(TextBlob("NLP is not great").sentiment)

# print(TextBlob("This is some positive text.").sentiment)


print(TextBlob("Amazon reports sales growth of 37%, topping estimates").sentiment)


print(TextBlob("Amazon posts biggest profit ever at height of pandemic in U.S.").sentiment)