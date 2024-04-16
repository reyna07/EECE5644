naive_bayes.py : Naive Bayes Classification by using nltk libraries

sklearn_train.py: Multinomial Naive Bayes Classification by using scikit-learn libraries

NOTES: 

1. For both python files, the download path must be modified as the path involving the python files.
2. sklearn_train.py also has a smoothener algorithm, first finding the best alpha minimizing the error, then training with the founded best alpha.
3. naive_bayes.py test accuracy is 97% whereas the best sklearn_train.py can do is 84%.
4. If i didn't smoothen in the sklearn_train.py, the testing accuracy was 82% with the default given alpha. 
