import numpy as np 

def lift(x): 

    l = len(x)
    index = l
    l_lifted = (l * (l+1) //2 ) + l
    x_new = np.zeros(l_lifted)
    
    x_new[:l] = x


    for i in range(l):
        for j in range(i+1):
            x_new[index] = x[i] * x[j]
            index += 1
    
    return x_new

x = [1, 2, 3, 4, 5]
x_lifted = lift(x)

print(x_lifted)


def liftDataset(X):
    n, d = X.shape
    d_lifted = d * (d+1) // 2 + d
    
    X_new = np.zeros((n, d_lifted))  
    
    for i in range(n):
        X_new[i] = lift(X[i])
    
    return X_new

x1 = np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]])
x_lifted1 = liftDataset(x1)

print(x_lifted1)
