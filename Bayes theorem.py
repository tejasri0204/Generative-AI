pip install GPyOpt numpy

import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt

# Define your generative model hyperparameter tuning function
def generative_model_hyperparameter_tuning(x):
    # Simulate a loss function (replace with your actual generative model evaluation)
    loss = np.sum(x**2)  # Example: sum of squared hyperparameters
    return -loss  # Negative because we're minimizing

# Define the search space for hyperparameters
hyperparameter_space = [{'name': 'hyperparam1', 'type': 'continuous', 'domain': (0, 1)},
                        {'name': 'hyperparam2', 'type': 'continuous', 'domain': (0, 1)},
                        # Add more hyperparameters here
                        ]

# Initialize Bayesian optimization
optimizer = BayesianOptimization(f=generative_model_hyperparameter_tuning, 
                                 domain=hyperparameter_space,
                                 initial_design_numdata=5)  

# Run Bayesian optimization
max_iterations = 10
optimizer.run_optimization(max_iter=max_iterations)

# Get best hyperparameters and the corresponding value
best_hyperparameters = optimizer.x_opt
best_value = -generative_model_hyperparameter_tuning(best_hyperparameters)  # Evaluate best hyperparameters

print("Best Hyperparameters:", best_hyperparameters)
print("Optimized Value:", best_value)

# Visualize the convergence plot manually using matplotlib
plt.plot(optimizer.Y_best)
plt.xlabel('Iteration')
plt.ylabel('Best Objective Value')
plt.title('Convergence Plot')
plt.show()