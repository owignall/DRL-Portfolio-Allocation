from transformers import pipeline
#, AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import torch.nn.functional as F


# with open("test_titles.txt", "r") as file:
#     text = file.read()
# titles = text.split("\n")

classifier = pipeline("sentiment-analysis")

# titles_slice = titles[50:60]

# results = classifier(titles_slice)

# for i in range(len(titles_slice)):
#     print(results[i], titles[i])


print(classifier(["Philip Morris loses case against Australia's tobacco plain"]))




