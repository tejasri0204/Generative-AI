#1-D

!pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate random data from a GMM
np.random.seed(0)

# Create a GMM with 2 components
n_samples = 300
n_components = 2

# Generate data from the GMM
gmm = GaussianMixture(n_components=n_components)

# Generate synthetic data for training the GMM (you can replace this with your actual data)
X = np.concatenate([np.random.normal(0, 1, n_samples // 2),
                    np.random.normal(5, 1, n_samples // 2)]).reshape(-1, 1)

# Fit the GMM to the data
gmm.fit(X)

# Generate samples from the trained GMM
X_generated, labels = gmm.sample(n_samples)

# Visualize the generated data
plt.figure(figsize=(10, 6))
plt.scatter(X_generated[:, 0], np.zeros_like(X_generated[:, 0]), c=labels, s=30, cmap='viridis')

plt.title('Generated Data from Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.yticks([])  # Hide y-axis ticks
plt.show()


#2-D

pip install numpy scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Define GMM parameters
num_samples = 300
num_features = 2
num_components = 3

# Define component parameters (mean and covariance)
means = np.array([[2, 2], [8, 8], [14, 14]])
covariances = np.array([[[1, 0.5], [0.5, 1]],
                        [[1, -0.7], [-0.7, 1]],
                        [[1, 0], [0, 1]]])

# Create and fit the GMM model
gmm = GaussianMixture(n_components=num_components, covariance_type='full')
gmm.fit(means)  # Fit the model to your provided means (you can use your actual data here)

# Generate random data using GMM
data = gmm.sample(num_samples)[0]

# Visualize generated data
plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data from GMM')
plt.show()

x, y = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 100),
                   np.linspace(data[:, 1].min(), data[:, 1].max(), 100))

# Evaluate the GMM components at each point
xy = np.column_stack([x.ravel(), y.ravel()])
Z = -gmm.score_samples(xy)
Z = Z.reshape(x.shape)

plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.7)
plt.contour(x, y, Z, levels=np.logspace(-3, 2, 10), cmap='Blues')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Components')
plt.show()


#3-D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

num_samples = 300
num_features = 3
num_components = 4

means = np.array([[2, 2, 2], [8, 8, 8], [14, 14, 14], [20, 20, 20]])
covariances = np.array([[[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]],
                        [[1, -0.7, 0.4], [-0.7, 1, -0.5], [0.4, -0.5, 1]],
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[1, 0.2, -0.3], [0.2, 1, 0.1], [-0.3, 0.1, 1]]])

gmm = GaussianMixture(n_components=num_components, covariance_type='full')
gmm.fit(means)

data = gmm.sample(num_samples)[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Generated 3D Data from GMM')
plt.show()
