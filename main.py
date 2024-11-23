import numpy as np
import matplotlib.pyplot as plt

# Generate input data (20 evenly spaced values between 0 and 1)
x = np.linspace(0.1, 1, 20).reshape(-1, 1)

# Generate target outputs using the given formula
y = ((1 + 0.6 * np.sin(2 * np.pi * x / 0.7)) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Normalize the data for better training stability
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())

# Define the MLP structure
input_size = 1
hidden_size = 6  # Choosing 6 hidden neurons (between 4 and 8)
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(output_size)

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Training using Backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = linear(final_input)

    # Compute error
    error = y - final_output

    # Backpropagation
    output_gradient = error * linear_derivative(final_output)
    hidden_error = np.dot(output_gradient, weights_hidden_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += learning_rate * np.dot(hidden_output.T, output_gradient)
    bias_output += learning_rate * np.sum(output_gradient, axis=0)
    weights_input_hidden += learning_rate * np.dot(x.T, hidden_gradient)
    bias_hidden += learning_rate * np.sum(hidden_gradient, axis=0)

# Predict the outputs using the trained MLP
predicted_output = linear(np.dot(sigmoid(np.dot(x, weights_input_hidden) + bias_hidden), weights_hidden_output) + bias_output)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Target", marker='o')
plt.plot(x, predicted_output, label="MLP Output", linestyle='dashed')
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.title("MLP Approximation of the Target Function")
plt.legend()
plt.grid()
plt.show()
