import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import confusion_matrix


# Specify the custom download path
custom_download_path = "C:/Users/ranab/Desktop/PhD_2/eece5644/project"

# Add the custom download path to NLTK's data path
nltk.data.path.insert(0, custom_download_path)

# Download NLTK resources if not already downloaded
if not os.path.exists(os.path.join(custom_download_path, 'corpora', 'movie_reviews')):
    nltk.download('movie_reviews', download_dir=custom_download_path)
if not os.path.exists(os.path.join(custom_download_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=custom_download_path)

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#Shuffle the documents
#random.shuffle(documents)

# Define feature extractor
def word_feats(words):
    return dict([(word, True) for word in words])

# Extract features from the dataset
featuresets = [(word_feats(words), category) for words, category in documents]

# Split the dataset into training and testing sets
train_set = featuresets[:1900]
test_set = featuresets[1900:]

X_train = [' '.join(words) for words, _ in train_set]
y_train = [category for _, category in train_set]

X_test = [' '.join(words) for words, _ in test_set]
y_test = [category for _, category in test_set]

# Vectorize input data
vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

logReg = LogisticRegression()
logReg.fit(X_train_vectorized, y_train)
y_pred = logReg.predict(X_test_vectorized)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: ", accuracy*100, "%")
confMat = confusion_matrix(y_pred, y_test)
print("Confusion Matrix: \n", confMat)
# Train logistic regression model


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

show_most_informative_features(vectorizer, logReg, n= 10)
