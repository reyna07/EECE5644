import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix
from lift import liftDataset 


# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40

psfx = postfix(N,d,sigma) 
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X = liftDataset(X)

print("Lifted dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

alphas = np.logspace(-10, 10, base = 2)
rmse_mean = []
rmse_std_dev = []

for alpha in alphas:
    model = Lasso(alpha=alpha)

    cv = KFold(n_splits=5, random_state=42,shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=cv,scoring="neg_root_mean_squared_error")

    rmse_mean.append(-np.mean(scores))
    rmse_std_dev.append(np.std(scores))
    print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha,rmse_mean[-1], rmse_std_dev[-1] ) )

optim_alpha = alphas[np.argmin(rmse_mean)]

lasso = Lasso(alpha=optim_alpha)
lasso.fit(X_train, y_train)

rmse_train = rmse(y_train, lasso.predict(X_train))
rmse_test = rmse(y_test, lasso.predict(X_test))

print("Optimal Alpha: ", optim_alpha, "Train RMSE: ", rmse_train, "Test RMSE: ", rmse_test)


print("Model parameters:")
print("\t Intercept: %3.5f" % lasso.intercept_,end="")
for i,val in enumerate(lasso.coef_):
    if np.abs(lasso.coef_[i]) > 1e-3:
        print(", β%d: %3.5f" % (i,val), end="")
print("\n")   

plt.errorbar(alphas, rmse_mean, yerr=rmse_std_dev, fmt="-o")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Cross-Validation Mean RMSE")
plt.title("Cross-Validation Mean RMSE vs. Alpha")
plt.grid(True)
plt.show()


