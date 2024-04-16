naive_bayes.py : Naive Bayes Classification by using nltk libraries

multi_naivebayes.py: Multinomial Naive Bayes Classification by using scikit-learn libraries

dataset: movie reviews dataset is used. -> https://www.kaggle.com/datasets/nltkdata/movie-review 

NOTES: 

1. For both python files, the download path for the dataset must be modified as the path involving the python files.
2. multi_naivebayes.py also has a smoothener algorithm, first finding the best alpha minimizing the error, then training with the founded best alpha.
3. naive_bayes.py test accuracy is 97% whereas the best multi_naivebayes.py can do is 84%.
4. If i didn't smoothen in the multi_naivebayes.py, the testing accuracy was 82% with the default given alpha.
5. Both files also give info about the best informative features. With nltk libraries, the result is very good, but scikit one just gives the most frequent words which are not very informative.  
