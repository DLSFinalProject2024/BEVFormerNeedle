# import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
sys.path.append("./")
from apps.models import ResNet9
from apps.simple_ml import train_cifar10, evaluate_cifar10
from needle.data import DataLoader, CIFAR10Dataset
import math
import matplotlib.pyplot as plt

def plot_figures(train_losses_ch4: [], train_losses_ch8: [], train_losses_ch16:[], train_acc_ch4: [], train_acc_ch8: [], train_acc_ch16: []):
    # Generate plots
    epochs = [i for i in range(1, 11)]

    train_acc_ch4 = [x*100 for x in train_acc_ch4]
    train_acc_ch8 = [x*100 for x in train_acc_ch8]
    train_acc_ch16 = [x*100 for x in train_acc_ch16]

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
    plt.savefig('training_loss_d2d.png')

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_ch4, label="ch4 Accuracy", marker="o")
    plt.plot(epochs, train_acc_ch8, label="ch8 Accuracy", marker="o")
    plt.plot(epochs, train_acc_ch16, label="ch16 Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy for ch4, ch8, and ch16")
    plt.legend()
    plt.grid()
    plt.savefig('training_accuracy_d2d.png')


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10Dataset(base_folder="./data/cifar-10-batches-py", train=True)
    test_dataset = CIFAR10Dataset(base_folder="./data/cifar-10-batches-py", train=False)
    print(f"len(train_dataset) = {len(train_dataset)}")
    print(f"len(test_dataset) = {len(test_dataset)}")

    # Create DataLoader instances
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
    device = ndl.cpu()

    # Define Deformable Vision Transformer model
    dattn_model_ch4 = nn.VisionTransformer(
        img_size=(32, 32),
        patch_size=4,
        in_channels=4,
        num_classes=10,
        embed_dim=64,
        num_blocks=1,
        num_heads=8,
        dim_head=8,
        mlp_hidden_dim=128,
        dropout=0.1,
        device=device,
        deform_attn_activate=True,
        dattn_dim_head=2,
        dattn_heads=2, 
        dattn_offset_groups=2
    )

    dattn_model_ch8 = nn.VisionTransformer(
        img_size=(32, 32),
        patch_size=4,
        in_channels=8,
        num_classes=10,
        embed_dim=64,
        num_blocks=1,
        num_heads=8,
        dim_head=8,
        mlp_hidden_dim=128,
        dropout=0.1,
        device=device,
        deform_attn_activate=True,
        dattn_dim_head=4, 
        dattn_heads=2, 
        dattn_offset_groups=2
    )

    dattn_model_ch16 = nn.VisionTransformer(
        img_size=(32, 32),
        patch_size=4,
        in_channels=16,
        num_classes=10,
        embed_dim=64,
        num_blocks=1,
        num_heads=8,
        dim_head=8,
        mlp_hidden_dim=128,
        dropout=0.1,
        device=device,
        deform_attn_activate=True,
        dattn_dim_head=8, 
        dattn_heads=2, 
        dattn_offset_groups=1
    )

    print(f"dattn_model_ch4.dattn.parmeters) = {sum([math.prod(each_w.shape) for each_w in dattn_model_ch4.dattn.fn.modules[0].parameters()])}")
    print(f"dattn_model_ch8.dattn.parmeters) = {sum([math.prod(each_w.shape) for each_w in dattn_model_ch8.dattn.fn.modules[0].parameters()])}")
    print(f"dattn_model_ch16.dattn.parmeters) = {sum([math.prod(each_w.shape) for each_w in dattn_model_ch16.dattn.fn.modules[0].parameters()])}")
    print(f"dattn_model_ch4.parmeters = {sum(math.prod(each_w.shape) for each_w in dattn_model_ch4.parameters())}")
    print(f"dattn_model_ch8.parmeters = {sum(math.prod(each_w.shape) for each_w in dattn_model_ch8.parameters())}")
    print(f"dattn_model_ch16.parmeters = {sum(math.prod(each_w.shape) for each_w in dattn_model_ch16.parameters())}")

    # Train the deformable attention model
    train_acc_dattn_ch4, train_losses_dattn_ch4 = train_cifar10(dattn_model_ch4, train_loader, n_epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001)
    train_acc_dattn_ch8, train_losses_dattn_ch8 = train_cifar10(dattn_model_ch8, train_loader, n_epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001)
    train_acc_dattn_ch16, train_losses_dattn_ch16 = train_cifar10(dattn_model_ch16, train_loader, n_epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001)

    # Evaluate the deformable attention model
    evaluate_cifar10(dattn_model_ch4, test_loader)
    evaluate_cifar10(dattn_model_ch8, test_loader)
    evaluate_cifar10(dattn_model_ch16, test_loader)

    '''
    train_losses_dattn_ch4 = [1.982, 1.648, 1.503, 1.410, 1.343, 1.292, 1.255, 1.228, 1.203, 1.183]
    train_losses_dattn_ch8 = [1.986, 1.619, 1.475, 1.388, 1.324, 1.276, 1.238, 1.203, 1.177, 1.154]
    train_losses_dattn_ch16 = [1.898, 1.536, 1.412, 1.331, 1.273, 1.232, 1.199, 1.173, 1.153, 1.134]

    train_acc_dattn_ch4 = [0.277, 0.390, 0.451, 0.486, 0.512, 0.531, 0.543, 0.554, 0.563, 0.569]
    train_acc_dattn_ch8 = [0.287, 0.407, 0.462, 0.496, 0.519, 0.539, 0.554, 0.567, 0.576, 0.585]
    train_acc_dattn_ch16 = [0.318, 0.438, 0.485, 0.515, 0.537, 0.554, 0.566, 0.574, 0.584, 0.591]
    '''

    # Plot
    plot_figures(train_losses_dattn_ch4, train_losses_dattn_ch8, train_losses_dattn_ch16, train_acc_dattn_ch4, train_acc_dattn_ch8, train_acc_dattn_ch16)