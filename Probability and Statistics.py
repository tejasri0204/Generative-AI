pip install numpy matplotlib scikit-learn

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data using GMM
num_samples = 300
true_means = np.array([[2, 2], [8, 3]])
true_covs = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]])
true_weights = np.array([0.6, 0.4])

# Generate samples
data = []
for i in range(num_samples):
    comp = np.random.choice(len(true_weights), p=true_weights)
    sample = np.random.multivariate_normal(true_means[comp], true_covs[comp])
    data.append(sample)

data = np.array(data)

# Fit a GMM to the data
num_components = 2
gmm = GaussianMixture(n_components=num_components, random_state=42)
gmm.fit(data)

# Generate samples from the learned GMM
generated_samples = gmm.sample(num_samples)
generated_samples = generated_samples[0]  # Extract generated samples

# Visualize the original data and generated data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], marker='o', color='blue', label='Original Data')
plt.title('Original Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], marker='x', color='red', label='Generated Data')
plt.title('Generated Data from GMM')
plt.legend()

plt.tight_layout()
plt.show()