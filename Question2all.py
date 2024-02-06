import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


#Question 2
#Define The parameters of the class-conditional Gaussian pdfs
m0 = np.array([-1, -1, -1, -1])
C0 = np.array([[5, 3, 1, -1],
               [3, 5, -2, -2],
               [1, -2, 6, 3],
               [-1, -2, 3, 4]])

m1 = np.array([1, 1, 1, 1])
C1 = np.array([[1.6, -0.5, -1.5, -1.2],
               [-0.5, 8, 6, -1.7],
               [-1.5, 6, 6, 0],
               [-1.2, -1.7, 0, 1.8]])

P_L0 = 0.35
P_L1 = 0.65

n_samples = 10000
samples_L0 = np.random.multivariate_normal(m0, C0, size=n_samples)
samples_L1 = np.random.multivariate_normal(m1, C1, size=n_samples)

#Compute the likelihoods
likelihood_ratio = []
def compute_likelihood(samples_L, x):
    for i in range(n_samples):
        likelihood_L0 = multivariate_normal.pdf(samples_L[i], mean=m0, cov=C0)
        likelihood_L1 = multivariate_normal.pdf(samples_L[i], mean=m1, cov=C1)
        if likelihood_L0 == 0:
            likelihood_ratio.append(x)
        else:
            likelihood_ratio.append(likelihood_L1 / likelihood_L0)
            
compute_likelihood(samples_L0, float('inf'))
compute_likelihood(samples_L1, 0)

sorted_likelihood_ratio = sorted(likelihood_ratio)

tpr_list = []
fpr_list = []
error_list = []

gamma_values = sorted_likelihood_ratio.copy()

#Classification
for gamma in gamma_values:
    #Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    true_positive = np.sum(np.array(likelihood_ratio[n_samples:]) > gamma)
    false_positive = np.sum(np.array(likelihood_ratio[:n_samples]) > gamma)
    true_negative = n_samples - false_positive
    false_negative = n_samples - true_positive

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    
    error = (P_L0 * false_positive + P_L1 * false_negative) / n_samples
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)
    error_list.append(error)

min_error_index = np.argmin(error_list)
min_error_gamma = gamma_values[min_error_index]

print(f"Minimum empirical error: {error_list[min_error_index]}")
print(f"Threshold Gamma for minimum emprical error: {min_error_gamma}")

min_error_tpr = tpr_list[min_error_index]
min_error_fpr = fpr_list[min_error_index]

print(f"TPR for minimum emprical error: {min_error_tpr}")
print(f"FPR for minimum emprical error: {min_error_fpr}\n")

#Plot ROC curve with TPR and FPR values attained by the minimum empirical error classifier
plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, color='blue', lw=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
plt.scatter(min_error_fpr, min_error_tpr, color='green', marker='o', s=100, label='Min Error Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve with Min Error Classifier')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('ROC_parta.png')

#Empirical Error Plot
plt.figure(figsize=(8, 6))
plt.plot(gamma_values, error_list, color='green', lw=2, label='Empirical Error')
plt.xscale('log')

#Plot the point where beta is 0.5
plt.scatter(gamma_values[min_error_index], error_list[min_error_index], color='red', marker='o', s=100, label='gamma for min EE')
plt.xlabel('Threshold (gamma)')
plt.ylabel('Empirical Error')
plt.title('Empirical Error vs. Threshold')
plt.grid(True)
plt.show()
#plt.savefig('EE_parta.png')

#PART B
beta_values = np.linspace(0, 1, 1000)

chernoff_bound_values = []

#Compute Chernoff bound for each beta value
for beta in beta_values:
    #Compute k(1/2) from equation 75
    diff_mu = m1 - m0
    mean_cov = (beta*C0 + (1-beta)*C1)
    ln_det_mean_cov = np.log(np.linalg.det(mean_cov))
    ln_det_sigma1 = (np.linalg.det(C0))**beta
    ln_det_sigma2 = (np.linalg.det(C1))**(1-beta)

    term1 = np.dot(diff_mu.T, np.linalg.inv(mean_cov).dot(diff_mu))
    term2 = 0.5 * np.log(np.linalg.det(mean_cov) /( ln_det_sigma1 * ln_det_sigma2))
    k_beta = 0.5 * beta*(1-beta) * term1 + term2

    # Compute Chernoff bound from equation 74
    chernoff_bound = (P_L0**(beta) * P_L1**(1-beta)) * np.exp(- k_beta)
    chernoff_bound_values.append(chernoff_bound)

#Find the beta that minimizes the Chernoff bound
min_beta_index = np.argmin(chernoff_bound_values)
min_chernoff_bound_beta = beta_values[min_beta_index]

math.exp(0)

#Chernoff bound curve
plt.plot(beta_values, chernoff_bound_values, color='blue', label='Chernoff Bound Curve')
plt.xlabel('Beta')
plt.ylabel('Chernoff Bound')
plt.title('Chernoff Bound Curve as a Function of Beta')
plt.grid(True)

# Plot the point where beta is minimum
plt.scatter(beta_values[min_beta_index], chernoff_bound_values[min_beta_index], color='red', marker='o', s=100, label='Min Chernoff Bound Beta')

# Plot the point where beta is 0.5
plt.scatter(0.5, chernoff_bound_values[int(len(chernoff_bound_values)/2)], color='green', marker='o', s=100, label='Beta = 0.5')
plt.legend()
plt.show()
#plt.savefig('chernoff.png')

print(f"Beta that minimizes the Chernoff bound: {min_chernoff_bound_beta}\n")

#PART C
#Diagonal covariance matrix
C0_diagonal = np.diag(np.diag(C0))
C1_diagonal = np.diag(np.diag(C1))

#Compute the likelihoods
likelihood_ratio_diagonal = []
for i in range(n_samples):
    likelihood_L0 = multivariate_normal.pdf(samples_L0[i], mean=m0, cov=C0_diagonal)
    likelihood_L1 = multivariate_normal.pdf(samples_L0[i], mean=m1, cov=C1_diagonal)
    if likelihood_L0 == 0:
        likelihood_ratio_diagonal.append(float('inf'))  # Assigning infinity for cases where likelihood_L0 is zero
    else:
        likelihood_ratio_diagonal.append(likelihood_L1 / likelihood_L0)

for i in range(n_samples):
    likelihood_L1 = multivariate_normal.pdf(samples_L1[i], mean=m1, cov=C1_diagonal)
    likelihood_L0 = multivariate_normal.pdf(samples_L1[i], mean=m0, cov=C0_diagonal)
    if likelihood_L0 == 0:
        likelihood_ratio_diagonal.append(0)  # Assigning 0 for cases where likelihood_L0 is zero
    else:
        likelihood_ratio_diagonal.append(likelihood_L1 / likelihood_L0)

sorted_likelihood_ratio_diagonal = sorted(likelihood_ratio_diagonal)

#Initialize lists
tpr_list_diagonal = []
fpr_list_diagonal = []
error_list_diagonal = []

for gamma in sorted_likelihood_ratio_diagonal:
    #calculate TPR and FPR
    true_positive_diagonal = np.sum(np.array(likelihood_ratio_diagonal[n_samples:]) > gamma)
    false_positive_diagonal = np.sum(np.array(likelihood_ratio_diagonal[:n_samples]) > gamma)
    true_negative_diagonal = n_samples - false_positive_diagonal
    false_negative_diagonal = n_samples - true_positive_diagonal

    tpr_diagonal = true_positive_diagonal / (true_positive_diagonal + false_negative_diagonal)
    fpr_diagonal = false_positive_diagonal / (false_positive_diagonal + true_negative_diagonal)

    error_diagonal = (P_L0 * false_positive_diagonal + P_L1 * false_negative_diagonal) / (2 * n_samples)
    
    tpr_list_diagonal.append(tpr_diagonal)
    fpr_list_diagonal.append(fpr_diagonal)

    error_list_diagonal.append(error_diagonal)

min_error_index_diagonal = np.argmin(error_list_diagonal)
min_error_gamma_diagonal = sorted_likelihood_ratio_diagonal[min_error_index_diagonal]

#Find the TPR and FPR values for the minimum empirical error
min_error_tpr_diagonal = tpr_list_diagonal[min_error_index_diagonal]
min_error_fpr_diagonal = fpr_list_diagonal[min_error_index_diagonal]

print(f"Minimum empirical error with model mismatch: {error_list_diagonal[min_error_index_diagonal]}")
print(f"Threshold Gamma for minimum emprical error with model mismatch: {min_error_gamma_diagonal}")


print(f"TPR for minimum emprical error with model mismatch: {min_error_tpr_diagonal}")
print(f"FPR for minimum emprical error with model mismatch: {min_error_fpr_diagonal}\n")

#Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_list_diagonal, tpr_list_diagonal, color='blue', lw=2, label='ROC Curve (Diagonal Covariance)')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
plt.scatter(min_error_fpr_diagonal, min_error_tpr_diagonal, color='green', marker='o', s=100, label='Min Error Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve with Diagonal Covariance')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('ROC_partc.png')


#Plot Empirical Error
plt.figure(figsize=(8, 6))
plt.plot(sorted_likelihood_ratio_diagonal, error_list_diagonal, color='green', lw=2, label='Empirical Error with Model Mismatch')
plt.xscale('log')
#Plot the point where beta is 0.5
plt.scatter(sorted_likelihood_ratio_diagonal[min_error_index_diagonal], error_list_diagonal[min_error_index_diagonal], color='red', marker='o', s=100, label='gamma for min EE with Model Mismatch')
plt.xlabel('Threshold (gamma)')
plt.ylabel('Empirical Error')
plt.title('Empirical Error with Model Mismatch vs. Threshold')
plt.grid(True)
plt.show()
#plt.savefig('EE_partc.png')


#PART D
#Initialize the sample covariance matrix
sample_covariance = np.zeros_like(C0, dtype=float)  


for sample in samples_L0:
    sample_diff = sample - m0
    sample_covariance += np.outer(sample_diff, sample_diff)

for sample in samples_L1:
    sample_diff = sample - m1
    sample_covariance += np.outer(sample_diff, sample_diff)

sample_covariance /= (2 * n_samples)

#compute likelihoods
likelihood_ratio_common_cov = []
for i in range(n_samples):
    likelihood_L0 = multivariate_normal.pdf(samples_L0[i], mean=m0, cov=sample_covariance)
    likelihood_L1 = multivariate_normal.pdf(samples_L0[i], mean=m1, cov=sample_covariance)
    if likelihood_L0 == 0:
        likelihood_ratio_common_cov.append(float('inf'))
    else:
        likelihood_ratio_common_cov.append(likelihood_L1 / likelihood_L0)

for i in range(n_samples):
    likelihood_L1 = multivariate_normal.pdf(samples_L1[i], mean=m1, cov=sample_covariance)
    likelihood_L0 = multivariate_normal.pdf(samples_L1[i], mean=m0, cov=sample_covariance)
    if likelihood_L0 == 0:
        likelihood_ratio_common_cov.append(0)
    else:
        likelihood_ratio_common_cov.append(likelihood_L1 / likelihood_L0)

sorted_likelihood_ratio_common_cov = sorted(likelihood_ratio_common_cov)

#Classification
tpr_list_common_cov = []
fpr_list_common_cov = []
error_list_common_cov = []

for gamma in sorted_likelihood_ratio_common_cov:
    #Calculate TPR and FPR
    true_positive_common_cov = np.sum(np.array(likelihood_ratio_common_cov[n_samples:]) > gamma)
    false_positive_common_cov = np.sum(np.array(likelihood_ratio_common_cov[:n_samples]) > gamma)
    true_negative_common_cov = n_samples - false_positive_common_cov
    false_negative_common_cov = n_samples - true_positive_common_cov

    tpr_common_cov = true_positive_common_cov / (true_positive_common_cov + false_negative_common_cov)
    fpr_common_cov = false_positive_common_cov / (false_positive_common_cov + true_negative_common_cov)

    tpr_list_common_cov.append(tpr_common_cov)
    fpr_list_common_cov.append(fpr_common_cov)

    error_common_cov = (P_L0 * false_positive_common_cov + P_L1 * false_negative_common_cov) / (2 * n_samples)  # Average error rate for both classes
    
    error_list_common_cov.append(error_common_cov)

min_error_index_common_cov = np.argmin(error_list_common_cov)
min_error_gamma_common_cov = sorted_likelihood_ratio_common_cov[min_error_index_common_cov]

#find TPR and FPR for the minimum empirical error
min_error_tpr_common_cov = tpr_list_common_cov[min_error_index_common_cov]
min_error_fpr_common_cov = fpr_list_common_cov[min_error_index_common_cov]

#Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_list_common_cov, tpr_list_common_cov, color='blue', lw=2, label='ROC Curve (Common Covariance)')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
plt.scatter(min_error_fpr_common_cov, min_error_tpr_common_cov, color='green', marker='o', s=100, label='Min Error Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve with Common Covariance')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('ROC_partd.png')


#Plot Empirical Error
plt.figure(figsize=(8, 6))
plt.plot(sorted_likelihood_ratio_common_cov, error_list_common_cov, color='green', lw=2, label='Empirical Error in Common Covariance Case')
plt.xscale('log')
#Plot the point where beta is 0.5
plt.scatter(sorted_likelihood_ratio_common_cov[min_error_index_common_cov], error_list_common_cov[min_error_index_common_cov], color='red', marker='o', s=100, label='gamma for min EE in Common Covariance Case')
plt.xlabel('Threshold (gamma)')
plt.ylabel('Empirical Error')
plt.title('Empirical Error in Common Covariance Case vs. Threshold')
plt.grid(True)
plt.show()
#plt.savefig('EE_partd.png')

# Print the minimum empirical error and the corresponding threshold (gamma)
print(f"Minimum empirical error with model mismatch: {error_list_common_cov[min_error_index_common_cov]}")
print(f"Threshold Gamma for minimum emprical error with model mismatch: {min_error_gamma_common_cov}")
print(f"TPR for minimum emprical error with model mismatch: {min_error_tpr_common_cov}")
print(f"FPR for minimum emprical error with model mismatch: {min_error_fpr_common_cov}\n")


#PART E
#cost matrix
B_values = np.linspace(0, 10, 100)

#Compute the expected risk
def expected_risk(gamma, B):
    Lambda = np.array([[0, B],
                       [1, 0]])
    likelihood_ratio = np.array([np.exp(np.log(P_L1/P_L0) - 0.5 * (np.dot((sample - m1), np.linalg.solve(C1, (sample - m1))) - np.dot((sample - m0), np.linalg.solve(C0, (sample - m0))))) for sample in np.concatenate((samples_L0, samples_L1))])
    decision = likelihood_ratio > gamma
    costs = np.array([Lambda[decision[i].astype(int)] for i in range(len(decision))])
    risk = np.mean(costs)
    return risk

min_expected_risk = np.zeros_like(B_values)
optimal_gamma = np.zeros_like(B_values)

#find the minimum expected risk
for i, B in enumerate(B_values):
    #initial guess for gamma
    if B < 1:
        initial_guess = 0.0  
    else:
        initial_guess = 1.0 

    result = minimize(expected_risk, initial_guess, args=(B,))
    optimal_gamma[i] = result.x[0]
    min_expected_risk[i] = result.fun

#Plot the minimum-risk decision boundary gamma and minimum expected risk
plt.figure(figsize=(10, 6))
plt.plot(B_values, optimal_gamma, label='Minimum-risk decision boundary ($\gamma$)')
plt.plot(B_values, min_expected_risk, label='Minimum expected risk')
plt.xlabel('B')
plt.ylabel('Value')
plt.title('Minimum-risk Decision Boundary and Minimum Expected Risk vs. B')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('Bfunctions.png')
