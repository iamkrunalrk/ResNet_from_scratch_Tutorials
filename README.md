# ResNet18 from Scratch - Notebook

## Overview
This notebook demonstrates how to implement **ResNet18** (Residual Networks) from scratch using **PyTorch**. ResNet is a deep convolutional neural network architecture that utilizes residual connections to ease the training of very deep networks. In this notebook, we will:

1. Build the ResNet18 architecture from scratch.
2. Train the model on a standard dataset (CIFAR-10).
3. Evaluate the performance of the model.

### Key Concepts
- **Residual Connections:** Skip connections that bypass one or more layers, allowing the network to learn residual mappings.
- **Batch Normalization:** A technique to normalize the output of each layer to improve training speed and stability.
- **Convolutional Neural Networks (CNNs):** A class of deep neural networks commonly used for visual tasks such as image classification and segmentation.

## Installation

Before running this notebook, ensure you have the following dependencies installed:

- Python
- PyTorch
- torchvision
- numpy

### Install Dependencies
To install the necessary packages, run the following command:

```bash
pip install torch torchvision matplotlib numpy
```

## Notebook Structure

### 1. **Import Libraries**
We start by importing the necessary libraries including PyTorch, torchvision, and matplotlib for visualization.

### 2. **ResNet18 Architecture**
The ResNet18 model is built from scratch by defining:
- The **BasicBlock** module, which consists of two convolutional layers with a skip connection.
- The **ResNet** class, which stacks the BasicBlocks along with other layers such as the initial convolution, fully connected layer, and pooling.

### 3. **Dataset Preparation**
The dataset (e.g., CIFAR-10) is loaded, preprocessed, and divided into training and validation sets. Data augmentation techniques like random flips and crops are applied to the training set.

### 4. **Model Training**
The model is trained on the dataset using an optimizer (e.g., Adam or SGD) and a loss function (e.g., CrossEntropyLoss). We track the model's loss and accuracy during training and adjust the learning rate if necessary.

### 5. **Model Evaluation**
The model's performance is evaluated on the validation set. We compute the accuracy and visualize some sample predictions.

## Conclusion

In this notebook, I have implemented the ResNet18 architecture from scratch, trained it on a dataset, and evaluated its performance. The use of residual connections has helped improve the model's accuracy and make it more feasible to train deeper networks. 

## Acknowledgments
- This project uses PyTorch and torchvision.
- The dataset used in this project is CIFAR-10.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
