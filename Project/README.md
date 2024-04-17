naive_bayes_nltk.py: Multinomial Naive Bayes Classification by using nltk libraries

multi_naivebayes_sklearn.py: Multinomial Naive Bayes Classification by using scikit-learn libraries - COULDNT FIND BETTER WAY TO SHOW INFORMATIVE FEATURES 

logistic_reg.py: Logistic Regression Classification by using scikit-learn libraries

svm.py: Support Vector Machines implementation by using scikit-learn libraries - DOES NOT WORK YET

dataset: movie reviews dataset is used. -> https://www.kaggle.com/datasets/nltkdata/movie-review 

NOTES: 

1. For each python file, the download path for the dataset must be modified as the path involving the python files.
2. multi_naivebayes_sklearn.py also has a smoothener algorithm, first finding the best alpha minimizing the error, and then training with the founded best alpha.
3. naive_bayes_nltk.py test accuracy is 97% whereas the best multi_naivebayes.py can do is 87%.
4. If I didn't smoothen in the multi_naivebayes.py, the testing accuracy was 82% with the default given alpha.
5. Accuracy for logistic regression algorithm is 87%.
6. Each file also gives info about the best informative features. With nltk libraries, the result is very good, but scikit one just gives the most frequent words which are not very informative.  
