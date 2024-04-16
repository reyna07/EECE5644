from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Prepare data
X = [' '.join(words) for words, _ in documents]
y = [category for _, category in documents]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Vectorize input data
vectorizer = CountVectorizer()
#vectorizer = CountVectorizer(min_df=1, tokenizer=word_tokenize)
# i tried using nltk tokenizer instead of default scikit
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
#best_features = best_classifr.feature_log_prob_


# Example review
review = "This movie is fantastic! I loved every moment of it."
# Vectorize the example review
example_review_vectorized = vectorizer.transform([review])

# Predict sentiment label
predicted_label = best_classifr.predict(example_review_vectorized)[0]

# Print the predicted sentiment label
print(f"Sentiment (Loaded Model): {predicted_label}")

# Get feature log probabilities
feature_log_probs = best_classifr.feature_log_prob_

# Get feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Print the most informative features for each class
for i, class_name in enumerate(best_classifr.classes_):
    print(f"\nMost informative features for class '{class_name}':")
    sorted_indices = feature_log_probs[i].argsort()[::-1]  # Sort indices by feature log probabilities
    top_features_indices = sorted_indices[:30]  # Get top 10 most informative features
    top_features = [feature_names[index] for index in top_features_indices]
    print(top_features)



