# Reimporting libraries after reset
import numpy as np
import matplotlib.pyplot as plt

# Constants
input_size = 1
output_size = 1
hidden_layer_size = 5  # Match your code's hidden layer size (5 neurons)
learning_rate = 0.01  # Match your learning rate
epochs = 10000  # Match your number of epochs

# Generate input and output data
X = np.linspace(0, 1, 20).reshape(-1, input_size)  # Input data
Y = (1 + 0.6 * np.sin(2 * np.pi * X[:, 0] / 0.7)) + 0.3 * np.sin(2 * np.pi * X[:, 0]) / 2  # Target function

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
w1 = np.random.rand(input_size, hidden_layer_size)  # Weights from input to hidden layer
b1 = np.random.rand(hidden_layer_size)  # Bias for hidden layer
w2 = np.random.rand(hidden_layer_size, output_size)  # Weights from hidden to output layer
b2 = np.random.rand(output_size)  # Bias for output layer

# Activation function: Hyperbolic Tangent (tanh)
def tanh(x):
    return np.tanh(x)

# Derivative of tanh
def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

# Training the model
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, w1) + b1  # Input to hidden layer
    hidden_layer_output = tanh(hidden_layer_input)  # Output of hidden layer

    output_layer_input = np.dot(hidden_layer_output, w2) + b2  # Input to output layer
    predicted_output = output_layer_input  # Linear activation at output layer

    # Error calculation
    error = Y.reshape(-1, 1) - predicted_output  # Error between target and predicted output

    # Report error every 5000 epochs
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

    # Backpropagation
    # Gradient for output layer
    d_output = error  # Linear activation gradient

    # Gradient for hidden layer
    error_hidden_layer = d_output.dot(w2.T) * tanh_derivative(hidden_layer_input)

    # Update weights and biases
    w2 += hidden_layer_output.T.dot(d_output) * learning_rate  # Hidden-to-output weights
    b2 += np.sum(d_output, axis=0) * learning_rate  # Output bias
    w1 += X.T.dot(error_hidden_layer) * learning_rate  # Input-to-hidden weights
    b1 += np.sum(error_hidden_layer, axis=0) * learning_rate  # Hidden bias

# Print final weights and biases
print("\nFinal Weights and Biases:")
print("Weights from Input to Hidden Layer (w1):", w1)
print("Biases for Hidden Layer (b1):", b1)
print("Weights from Hidden to Output Layer (w2):", w2)
print("Biases for Output Layer (b2):", b2)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X, Y, label='True Output', color='g', marker='o')
plt.plot(X, predicted_output, label='Predicted Output', color='r', linestyle='--', marker='x')
plt.title('Predicted Output vs True Output')
plt.xlabel('Input X')
plt.ylabel('Output Y')
plt.legend()
plt.grid(True)
plt.show()
