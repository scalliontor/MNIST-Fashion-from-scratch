"""
Fashion MNIST Classification from Scratch
Neural Network Implementation without TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import gzip
import struct
from urllib.request import urlopen
import os

np.random.seed(42)

def download_fashion_mnist():
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    data_dir = 'fashion_mnist_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f'Downloading {file}...')
            with urlopen(base_url + file) as response:
                with open(file_path, 'wb') as f:
                    f.write(response.read())
    
    return data_dir

def load_fashion_mnist_data(data_dir):
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num_images, rows, cols)
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    train_images = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    return train_images, train_labels, test_images, test_labels

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

def calculate_accuracy(predictions, y_true):
    predicted_classes = np.argmax(predictions, axis=1)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(predicted_classes == y_true)

def create_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

def main():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("Loading Fashion MNIST dataset...")
    data_dir = download_fashion_mnist()
    X_train_full, y_train_full, X_test, y_test = load_fashion_mnist_data(data_dir)
    
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    
    print(f"Training: {X_train.shape}, Validation: {X_valid.shape}, Test: {X_test.shape}")
    
    dense1 = Layer_Dense(784, 128)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(128, 64)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(64, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-4)
    
    epochs = 50
    batch_size = 128
    print_every = 5
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    print("Training started...")
    for epoch in range(epochs):
        epoch_loss, epoch_accuracy, n_batches = 0, 0, 0
        
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dense3.forward(activation2.output)
            loss = loss_activation.forward(dense3.output, y_batch)
            
            accuracy = calculate_accuracy(loss_activation.output, y_batch)
            epoch_loss += loss
            epoch_accuracy += accuracy
            n_batches += 1
            
            loss_activation.backward(loss_activation.output, y_batch)
            dense3.backward(loss_activation.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)
            
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
            optimizer.post_update_params()
        
        avg_train_loss = epoch_loss / n_batches
        avg_train_accuracy = epoch_accuracy / n_batches
        
        dense1.forward(X_valid)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        val_loss = loss_activation.forward(dense3.output, y_valid)
        val_accuracy = calculate_accuracy(loss_activation.output, y_valid)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    test_loss = loss_activation.forward(dense3.output, y_test)
    test_accuracy = calculate_accuracy(loss_activation.output, y_test)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    test_predictions = np.argmax(loss_activation.output, axis=1)
    
    n_samples = 20
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    plt.figure(figsize=(16, 8))
    for i, idx in enumerate(indices):
        plt.subplot(4, 5, i + 1)
        image = X_test[idx].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        
        predicted_label = class_names[test_predictions[idx]]
        actual_label = class_names[y_test[idx]]
        color = 'green' if test_predictions[idx] == y_test[idx] else 'red'
        
        plt.title(f'Pred: {predicted_label}\nActual: {actual_label}', 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
