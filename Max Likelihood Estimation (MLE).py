pip install numpy scipy matplotlib

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate some sample data from a Gaussian distribution
np.random.seed(42)
true_mean = 2.0
true_std = 1.5
sample_size = 100
data = np.random.normal(true_mean, true_std, sample_size)

# Define the log-likelihood function for a Gaussian distribution
def log_likelihood(params, data):
    mean, std = params
    log_likelihoods = norm.logpdf(data, mean, std)
    return np.sum(log_likelihoods)

# Initial guess for mean and standard deviation
initial_guess = [0.0, 1.0]

# Optimize the log-likelihood function to find MLE estimates
from scipy.optimize import minimize
result = minimize(lambda params: -log_likelihood(params, data), initial_guess)

# Extract the MLE estimates
estimated_mean, estimated_std = result.x

# Print the results
print(f"True Mean: {true_mean:.2f}, Estimated Mean: {estimated_mean:.2f}")
print(f"True Std: {true_std:.2f}, Estimated Std: {estimated_std:.2f}")

# Plot the histogram of the data and the estimated Gaussian distribution
plt.hist(data, bins=20, density=True, alpha=0.6, label="Data")
x_range = np.linspace(data.min(), data.max(), 100)
plt.plot(x_range, norm.pdf(x_range, estimated_mean, estimated_std), 'r',
         label="Estimated Gaussian")
plt.legend()
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("MLE of Gaussian Distribution Parameters")
plt.show()