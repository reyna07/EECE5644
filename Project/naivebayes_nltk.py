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


# split the dataset into training and testing sets
train_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy:.4f}")

# save the classifier to a file
model_file = "naive_bayes_classifier.pickle"
with open(model_file, 'wb') as f:
    pickle.dump(classifier, f)


# example review
review = "This movie is fantastic! I loved every moment of it."
#review = "güzel bir filmdi."

words = word_tokenize(review)
features = word_feats(words)


loaded_sentiment = classifier.classify(features)
print("Sentiment (Loaded Model):", loaded_sentiment)

def show_most_informative_features(classifier, n=10):
        positives = []
        negatives = []
        pos_values = []
        neg_values = []
        # Determine the most relevant features, and display them.
        cpdist = classifier._feature_probdist
        print('Most Informative Features')

        for (fname, fval) in classifier.most_informative_features(n+30):
            def labelprob(l):
                return cpdist[l, fname].prob(fval)

            labels = sorted([l for l in classifier._labels
                             if fval in cpdist[l, fname].samples()],
                            key=labelprob)
            if len(labels) == 1:
                continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0, fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
            if l1 == "pos": 
                positives.append(fname)
                pos_values.append(cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
            else:
                negatives.append(fname)
                neg_values.append(cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
        for i in range(10):
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % ( neg_values[i], negatives[i], pos_values[i], positives[i]))

# the most informative features
show_most_informative_features(classifier, 20)




