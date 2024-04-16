import nltk
import pickle
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import os
import numpy as np


# Specify the custom download path
custom_download_path = "C:/Users/ranab/Desktop/PhD_2/eece5644/project"

# Add the custom download path to NLTK's data path
nltk.data.path.insert(0, custom_download_path)

# Download NLTK resources if not already downloaded
if not os.path.exists(os.path.join(custom_download_path, 'corpora', 'movie_reviews')):
    nltk.download('movie_reviews', download_dir=custom_download_path)
if not os.path.exists(os.path.join(custom_download_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=custom_download_path)

# Now you can download the NLTK resources
#nltk.download('movie_reviews', download_dir=custom_download_path)
#nltk.download('punkt', download_dir=custom_download_path)


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Define feature extractor
def word_feats(words):
    return dict([(word, True) for word in words])

# Extract features from the dataset
featuresets = [(word_feats(words), category) for words, category in documents]

# Split the dataset into training and testing sets
train_set = featuresets[:1900]
test_set = featuresets[1900:]

# Train the classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy:.4f}")

# Save the classifier to a file
model_file = "naive_bayes_classifier.pickle"
with open(model_file, 'wb') as f:
    pickle.dump(classifier, f)


# Example review
review = "This movie is fantastic! I loved every moment of it."
#review = "güzel bir filmdi."
# Tokenize the review
words = word_tokenize(review)

# Extract features from the review
features = word_feats(words)

# Example usage of the loaded classifier
loaded_sentiment = classifier.classify(features)
print("Sentiment (Loaded Model):", loaded_sentiment)

# Extract the most informative features
classifier.show_most_informative_features(20)

