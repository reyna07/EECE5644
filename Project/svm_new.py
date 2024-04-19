import nltk
from nltk.corpus import movie_reviews
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.svm import SVC
import os

# specify the download path
download_path = "C:/Users/ranab/Desktop/PhD_2/eece5644/project"

# add the custom download path to NLTK's data path
nltk.data.path.insert(0, download_path)

# download NLTK resources if not already downloaded
if not os.path.exists(os.path.join(download_path, 'corpora', 'movie_reviews')):
    nltk.download('movie_reviews', download_dir=download_path)
if not os.path.exists(os.path.join(download_path, 'corpora', 'stopwords')):
    nltk.download('stopwords', download_dir=download_path)
if not os.path.exists(os.path.join(download_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=download_path)


# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


def word_feats(words): # feature extractor
    return dict([(word, True) for word in words])

featuresets = [(word_feats(words), category) for words, category in documents]# extracting features from the dataset

# Prepare data
X = [' '.join(words) for words, _ in documents]
y = [category for _, category in documents]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Linear SVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2))),
    ('svc', LinearSVC(random_state=42, dual=False))
])

# Parameter grid for GridSearch
param_grid = {
    'tfidf__max_features': [1000, 2000, 3000],
    'tfidf__sublinear_tf': [True, False],
    'svc__C': [0.1, 1, 10]
}

# Grid search to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("-------------------------------------------------------------------\n")
print("Support Vector Machines(SVM) Accuracy and Best Informative Features\n")
print("-------------------------------------------------------------------\n")       



# Best estimator from grid search
best_classifier = grid_search.best_estimator_
#print(best_classifier)
# Predict and calculate accuracy
y_pred = best_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Best Parameters:", grid_search.best_params_)


print("-------------------------------------------------------------------\n")
# Function to find most informative features for Linear SVC
def most_informative_features(vectorizer, classifier, n=10):
    feature_names = vectorizer.get_feature_names_out()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    print("Most Informative Features:")
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (-coef_1, fn_1, coef_2, fn_2))

# Display most informative features
vectorizer = best_classifier.named_steps['tfidf']
classifier = best_classifier.named_steps['svc']
most_informative_features(vectorizer, classifier)


print("-------------------------------------------------------------------\n")

# Optional: Example review classification
review = "This movie is fantastic! I loved every moment of it."
review_vectorized = vectorizer.transform([review])
predicted_sentiment = classifier.predict(review_vectorized)
print(f"Sentiment (Loaded Model): {predicted_sentiment[0]}")


print("-------------------------------------------------------------------\n")




