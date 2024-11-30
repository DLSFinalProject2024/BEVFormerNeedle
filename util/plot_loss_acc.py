 
from typing import Optional
import json
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate plots
    epochs = [i for i in range(1, 11)]
    train_losses_a = [1.903, 1.442, 1.318, 1.246, 1.195, 1.155, 1.123, 1.096, 1.071, 1.048]
    train_losses_b = [2.002, 1.618, 1.459, 1.379, 1.330, 1.289, 1.262, 1.239, 1.219, 1.202]
    train_acc_a = [0.312, 0.474, 0.522, 0.549, 0.568, 0.584, 0.595, 0.606, 0.613, 0.621]
    train_acc_b = [0.277, 0.409, 0.471, 0.502, 0.519, 0.535, 0.544, 0.554, 0.562, 0.568]

    train_acc_a = [x*100 for x in train_acc_a]
    train_acc_b = [x*100 for x in train_acc_b]

    # Plot Training Loss of (a) and (b0)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses_a, label=f'Training Loss of model (a) Baseline')
    plt.plot(epochs, train_losses_b, label=f'Training Loss of model (b) Deformable Attention')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss along Different Epochs on model Baseline and Deformable Attention.')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Training Accuracy of (a) and (b)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_a, label=f'Training Accuracy of model (a) Baseline')
    plt.plot(epochs, train_acc_b, label=f'Training Accuracy of model (b) Deformable Attention')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy along Different Epochs on model Baseline and Deformable Attention.')
    plt.legend()
    plt.grid()
    plt.show()
