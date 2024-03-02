import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix
from lift import liftDataset 


# Number of samples
N = 2000

# Noise variance 
sigma = 0.01

# Feature dimension
d = 40

psfx = postfix(N,d,sigma) 
      
X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X = liftDataset(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))


fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rmse_train = []
rmse_test = []

for fr in fractions:

    n_samples = int(fr * X_train.shape[0])
    
    X_train_subset = X_train[:n_samples]
    y_train_subset = y_train[:n_samples]

    model = LinearRegression()
    print("Fitting linear model...",end="")
    model.fit(X_train_subset, y_train_subset)
    print(" done")

    rmse_train.append(rmse(y_train_subset,model.predict(X_train_subset)))
    rmse_test.append(rmse(y_test,model.predict(X_test)))

    print("Train RMSE = %f, Test RMSE = %f" % (rmse_train[-1],rmse_test[-1]))

    print("Model parameters:")
    print("\t Intercept: %3.5f" % model.intercept_,end="")
    for i,val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i,val), end="")
    print("\n") 

fractions_b = np.dot(fractions, X_train.shape[0]) 
fractions_c = np.dot(fractions, X_test.shape[0])

plt.figure(figsize=(10, 6))
plt.plot(fractions_b, rmse_train, label='Train RMSE')
plt.title('Train RMSE vs. Number of Training Samples')
plt.xlabel('Number of Training Samples')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()    

plt.figure(figsize=(10, 6))
plt.plot(fractions_c, rmse_test, label='Test RMSE')
plt.title('Test RMSE vs. Number of Test Samples')
plt.xlabel('Number of Test Samples')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()    






