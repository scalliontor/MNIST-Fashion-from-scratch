# Fashion MNIST Classification from Scratch

A neural network implementation built from scratch in Python to classify Fashion MNIST dataset without using TensorFlow, PyTorch, or any other deep learning frameworks.

## Overview

This project demonstrates the fundamental concepts of neural networks by implementing:
- Dense (fully connected) layers
- ReLU activation function
- Softmax activation function  
- Categorical cross-entropy loss
- Adam optimizer with learning rate decay
- Backpropagation algorithm

## Architecture

```
Input Layer:     784 neurons (28×28 flattened images)
Hidden Layer 1:  128 neurons + ReLU activation
Hidden Layer 2:  64 neurons + ReLU activation
Output Layer:    10 neurons + Softmax activation
```

## Dataset

Fashion MNIST consists of 70,000 grayscale images (60,000 training, 10,000 test) of 28×28 pixels, each belonging to one of 10 clothing categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Installation

1. Clone the repository:
```bash
git clone https://github.com/scalliontor/MNIST-Fashion-from-scratch.git
cd MNIST-Fashion-from-scratch
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Python script:
```bash
python fashion_mnist_clean.py
```

### Or use the Jupyter notebook:
```bash
jupyter notebook Fashion_MNIST_From_Scratch.ipynb
```

## Features

- **Automatic dataset download**: Downloads Fashion MNIST data automatically
- **Data preprocessing**: Normalizes pixel values and handles train/validation split
- **Mini-batch training**: Supports configurable batch sizes
- **Training visualization**: Plots loss and accuracy curves
- **Test evaluation**: Evaluates model performance on test set
- **Prediction visualization**: Shows sample predictions with correct/incorrect labels

## Results

The neural network achieves competitive accuracy on the Fashion MNIST dataset, demonstrating that fundamental neural network concepts can be implemented effectively from scratch.

## Implementation Details

### Core Components

- **Layer_Dense**: Fully connected layer with forward and backward propagation
- **Activation_ReLU**: ReLU activation function with gradient computation
- **Activation_Softmax**: Softmax activation for multi-class classification
- **Loss_CategoricalCrossentropy**: Cross-entropy loss function
- **Optimizer_Adam**: Adam optimizer with momentum and adaptive learning rates

### Training Process

1. Forward propagation through all layers
2. Loss calculation using cross-entropy
3. Backward propagation to compute gradients
4. Parameter updates using Adam optimizer
5. Validation on separate dataset

## Dependencies

- numpy: Numerical computations
- matplotlib: Plotting and visualization
- gzip, struct, urllib: Dataset downloading and processing

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.
