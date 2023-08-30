#Gaussian distribution

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Gaussian distribution
mean = 0
std_dev = 1
sample_size = 1000

# Generate random samples from the Gaussian distribution
samples = np.random.normal(mean, std_dev, sample_size)

# Plotting the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# Plotting the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# Plotting the probability density function (PDF)
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
pdf = (1/(std_dev * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mean)/std_dev)**2)
plt.plot(x, pdf, color='r')

plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend(['PDF', 'Histogram'])
plt.show()


# Bernoulli distribution

# Parameters for the Bernoulli distribution
p = 0.3  # Probability of success (e.g., getting a '1')
sample_size = 1000

# Generate random samples from the Bernoulli distribution
samples = np.random.binomial(1, p, sample_size)

# Plotting the histogram of the samples
plt.hist(samples, bins=[0, 1, 2], density=True, alpha=0.6, color='b')

plt.title('Bernoulli Distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.xticks([0, 1], ['0', '1'])
plt.show()


# Categorical distribution

# Parameters for the categorical distribution
categories = ['A', 'B', 'C', 'D']
probabilities = [0.2, 0.3, 0.1, 0.4]
sample_size = 1000

# Generate random samples from the categorical distribution
samples = np.random.choice(categories, size=sample_size, p=probabilities)

# Counting the occurrences of each category
unique, counts = np.unique(samples, return_counts=True)

# Plotting the bar chart of category occurrences
plt.bar(unique, counts/sample_size, color='y', alpha=0.6)

plt.title('Categorical Distribution')
plt.xlabel('Category')
plt.ylabel('Probability')
plt.show()


# Exponential distribution

# Parameters for the exponential distribution
scale = 2.0  # Inverse of the rate parameter (lambda)
sample_size = 1000

# Generate random samples from the exponential distribution
samples = np.random.exponential(scale, sample_size)

# Plotting the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='m')

# Plotting the probability density function (PDF)
x = np.linspace(0, 10, 100)
pdf = (1/scale) * np.exp(-x/scale)
plt.plot(x, pdf, color='r')

plt.title('Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend(['PDF', 'Histogram'])
plt.show()

