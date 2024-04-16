from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import movie_reviews
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk

# specify the download path
download_path = "C:/Users/ranab/Desktop/PhD_2/eece5644/project"

# add the custom download path to NLTK's data path
nltk.data.path.insert(0, download_path)

# download NLTK resources if not already downloaded
if not os.path.exists(os.path.join(download_path, 'corpora', 'movie_reviews')):
    nltk.download('movie_reviews', download_dir=download_path)
if not os.path.exists(os.path.join(download_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=download_path)

# load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# prepare data
X = [' '.join(words) for words, _ in documents]
y = [category for _, category in documents]

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Vectorize input data
vectorizer = CountVectorizer()
#vectorizer = CountVectorizer(min_df=1, tokenizer=word_tokenize)
# tried using nltk tokenizer instead of default scikit
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


alpha_values = np.logspace(-15, 5, base=2) #range of alpha

accuracies = []

for i in range(10): #repeating for 10 times
    
    accuracies_for_iteration = []
    
    for alpha in alpha_values:
        classifr = MultinomialNB(alpha=alpha)
        classifr.fit(X_train_vectorized, y_train) #fitting
        accuracy = classifr.score(X_test_vectorized, y_test) #directly testing
        accuracies_for_iteration.append(accuracy)

    accuracies.append(accuracies_for_iteration)

accuracies = np.array(accuracies)
avg_accuracy = np.mean(accuracies, axis=0) #taking the mean
std_deviation = np.std(accuracies, axis=0) #taking the standard deviation


best_alpha_index = np.argmax(avg_accuracy) #best accuracy index
best_alpha = alpha_values[best_alpha_index] #best alpha


print(f"Best alpha value: {best_alpha}")
print(f"Maximum average accuracy: {avg_accuracy[best_alpha_index]}")

plt.errorbar(alpha_values, avg_accuracy, yerr=std_deviation, fmt='-o', capsize=5) #error plot for std deviation and mean
plt.xlabel('Alpha')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy as a Function of Alpha')
plt.xscale('log')
plt.grid(True)
plt.show()


best_classifr = MultinomialNB(alpha=best_alpha) #best classifier
best_classifr.fit(X_train_vectorized, y_train)



# an example review
review = "This movie is fantastic! I loved every moment of it."
example_review_vectorized = vectorizer.transform([review])

# predict and print sentiment label
predicted_label = best_classifr.predict(example_review_vectorized)[0]
print(f"Sentiment (Loaded Model): {predicted_label}")

# feature log probabilities
feature_log_probs = best_classifr.feature_log_prob_

#feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Print the most informative features for each class
for i, class_name in enumerate(best_classifr.classes_):
    print(f"\nMost informative features for class '{class_name}':")
    sorted_indices = feature_log_probs[i].argsort()[::-1]  
    top_features_indices = sorted_indices[:20]  #top 10 most informative features
    top_features = [feature_names[index] for index in top_features_indices]
    print(top_features)



