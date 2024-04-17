from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
import pickle
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import os
import numpy as np

# specify the download path
download_path = "C:/Users/ranab/Desktop/PhD_2/eece5644/project"

# add the custom download path to NLTK's data path
nltk.data.path.insert(0, download_path)

# download NLTK resources if not already downloaded
if not os.path.exists(os.path.join(download_path, 'corpora', 'movie_reviews')):
    nltk.download('movie_reviews', download_dir=download_path)
if not os.path.exists(os.path.join(download_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=download_path)


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

def word_feats(words): # feature extractor
    return dict([(word, True) for word in words])

featuresets = [(word_feats(words), category) for words, category in documents]# extracting features from the dataset

train_set = featuresets[:1900]
test_set = featuresets[1900:]

classif = SklearnClassifier(LinearSVC())
classif.train(train_set)

accuracy = nltk.classify.util.accuracy(classif, test_set)
print(f"Accuracy: {accuracy:.4f}")


# Example review
review = "This movie is fantastic! I loved every moment of it."
#review = "güzel bir filmdi."
# Tokenize the review
words = word_tokenize(review)

# Extract features from the review
features = word_feats(words)

# Example usage of the loaded classifier
loaded_sentiment = classif.classify(features)
print("Sentiment (Loaded Model):", loaded_sentiment)

# Extract the most informative features
print("Most Informative Features:")
classif.__getattribute__


