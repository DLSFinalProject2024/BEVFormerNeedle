from typing import Optional
import json
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate plots
    epochs = [i for i in range(1, 11)]

    train_losses_a = [1.903, 1.442, 1.318, 1.246, 1.195, 1.155, 1.123, 1.096, 1.071, 1.048]
    train_losses_b = [1.986, 1.619, 1.475, 1.388, 1.324, 1.276, 1.238, 1.203, 1.177, 1.154]
    train_acc_a = [0.312, 0.474, 0.522, 0.549, 0.568, 0.584, 0.595, 0.606, 0.613, 0.621]
    train_acc_b = [0.287, 0.407, 0.462, 0.496, 0.519, 0.539, 0.554, 0.567, 0.576, 0.585]

    train_acc_a = [x*100 for x in train_acc_a]
    train_acc_b = [x*100 for x in train_acc_b]

    train_losses_ch4 = [1.982, 1.648, 1.503, 1.410, 1.343, 1.292, 1.255, 1.228, 1.203, 1.183]
    train_losses_ch8 = [1.986, 1.619, 1.475, 1.388, 1.324, 1.276, 1.238, 1.203, 1.177, 1.154]
    train_losses_ch16 = [1.898, 1.536, 1.412, 1.331, 1.273, 1.232, 1.199, 1.173, 1.153, 1.134]

    train_acc_ch4 = [0.277, 0.390, 0.451, 0.486, 0.512, 0.531, 0.543, 0.554, 0.563, 0.569]
    train_acc_ch8 = [0.287, 0.407, 0.462, 0.496, 0.519, 0.539, 0.554, 0.567, 0.576, 0.585]
    train_acc_ch16 = [0.318, 0.438, 0.485, 0.515, 0.537, 0.554, 0.566, 0.574, 0.584, 0.591]

    test_losses_ch4 = 1.240
    test_losses_ch8 = 1.183
    test_losses_ch16 = 1.174

    test_acc_ch4 = 0.553
    test_acc_ch8 = 0.574
    test_acc_ch16 = 0.576

    parameters_num_ch4 = 126
    parameters_num_ch8 = 334
    parameters_num_ch16 = 1002

    elapsed_time_s_it_ch4 = 3.67
    elapsed_time_s_it_ch8 = 4.87
    elapsed_time_s_it_ch16 = 6.89

    # Plot Training Loss of (a) and (b0)
    '''
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
    '''

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses_ch4, label="ch4 Loss", marker="o")
    plt.plot(epochs, train_losses_ch8, label="ch8 Loss", marker="o")
    plt.plot(epochs, train_losses_ch16, label="ch16 Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss for ch4, ch8, and ch16")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_ch4, label="ch4 Accuracy", marker="o")
    plt.plot(epochs, train_acc_ch8, label="ch8 Accuracy", marker="o")
    plt.plot(epochs, train_acc_ch16, label="ch16 Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy for ch4, ch8, and ch16")
    plt.legend()
    plt.grid()
    plt.show()

