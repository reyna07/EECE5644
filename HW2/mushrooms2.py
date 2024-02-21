import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
import matplotlib.pyplot as plt

#modified version of first case since we need to consider missing categories
X = pd.read_csv('X_msrm.csv') #read the data file
y = pd.read_csv('y_msrm.csv')


y = y.values.ravel() #transpose the y matrix


def train_naive_bayes(test_percentage):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)
    min_cats = []
    for col in X.columns:
        min_cats.append(len(X[col].unique()))

    alpha_values = np.logspace(-15, 5, base=2)  #range of alpha
    results = []
    
    for alpha in alpha_values:
        classifr = CategoricalNB(alpha=alpha, min_categories=min_cats)
        classifr.fit(X_train, y_train) #fitting
        y_pred_prob = classifr.predict_proba(X_test)[:, 1] #prediction probability
        y_pred =  classifr.predict(X_test) 
        roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred)
        results.append({'alpha': alpha, 'roc_auc': roc_auc, 'accuracy': accuracy, 'f1_score': f1_score})
    
    results_frame = pd.DataFrame(results)

    plt.plot(results_frame['alpha'], results_frame['roc_auc'], label='ROC AUC')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Performance')
    plt.title('ROC AUC')
    plt.show()

    plt.plot(results_frame['alpha'], results_frame['accuracy'], label='Accuracy')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Performance')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(results_frame['alpha'], results_frame['f1_score'], label='F1 Score')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Performance')
    plt.title('F1 Score')
    plt.legend()
    plt.show()

    plt.plot(results_frame['alpha'], results_frame['roc_auc'], label='ROC AUC')
    plt.plot(results_frame['alpha'], results_frame['accuracy'], label='Accuracy')
    plt.plot(results_frame['alpha'], results_frame['f1_score'], label='F1 Score')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Performance')
    plt.title('Graph of the Predictive Performance of the Trained Classifier')
    plt.legend()
    plt.show()

    best_index = results_frame['roc_auc'].idxmax() #index of the best alpha
    best_alpha = results_frame.loc[best_index]['alpha'] #best alpha

    print(f"Best alpha value: {best_alpha}")
    print(f"Maximized ROC AUC: {results_frame['roc_auc'][best_index]}") #best alpha max value


    best_classifr = CategoricalNB(alpha=best_alpha) #best classifier
    best_classifr.fit(X_train, y_train)
    best_feature = best_classifr.feature_log_prob_


    print("Parameters for the best alpha value:") # the best classifier parameters
    print(best_feature)
   
    
#train_naive_bayes(0.2) - actually I could ve created this code earlier for the first part as well 
train_naive_bayes(0.99)



