import numpy as np
import matplotlib.pyplot as plt

# Simulate vessel lengths and predictions
np.random.seed(42)
N = 100  # Number of vessels
l_max = 300  # Upper bound for vessel lengths

# Generate actual lengths and predictions
actual_lengths = np.random.uniform(50, l_max, N)
predicted_lengths = actual_lengths + np.random.normal(0, 30, N)  # Add some noise

# Calculate Percentage Length Accuracy (PLA)
normalized_errors = np.abs(np.minimum(predicted_lengths, l_max) - np.minimum(actual_lengths, l_max)) / np.minimum(actual_lengths, l_max)
mean_normalized_error = np.mean(normalized_errors)
pla = 1 - min(mean_normalized_error, 1)

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(actual_lengths, predicted_lengths, c=normalized_errors, cmap='coolwarm', edgecolor='k')
plt.colorbar(label='Normalized Error')
plt.plot([0, l_max], [0, l_max], 'k--', label='Perfect Prediction')
plt.xlabel('Actual Lengths')
plt.ylabel('Predicted Lengths')
plt.title(f'Vessel Length Predictions (PLA = {pla:.3f})')
plt.legend()
plt.show()

