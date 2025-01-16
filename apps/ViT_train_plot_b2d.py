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

def plot_figures(train_losses_base: [], train_losses_dattn: [], train_acc_base: [], train_acc_dattn: []):
    # Generate plots
    epochs = [i for i in range(1, 11)]

    train_acc_base = [x*100 for x in train_acc_base]
    train_acc_dattn = [x*100 for x in train_acc_dattn]

    # Plot Training Loss of (a) and (b0)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses_base, label=f'Training Loss of model (a) Baseline')
    plt.plot(epochs, train_losses_dattn, label=f'Training Loss of model (b) Deformable Attention')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss along Different Epochs on model Baseline and Deformable Attention.')
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig('training_loss_b2d.png')

    # Plot Training Accuracy of (a) and (b)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_base, label=f'Training Accuracy of model (a) Baseline')
    plt.plot(epochs, train_acc_dattn, label=f'Training Accuracy of model (b) Deformable Attention')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy along Different Epochs on model Baseline and Deformable Attention.')
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig('training_accuracy_b2d.png')


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
    in_channels_base = 3
    in_channels_dattn = 8
    # Define Vision Transformer model
    base_model = nn.VisionTransformer(
        img_size=(32, 32),
        patch_size=4,
        in_channels=in_channels_base,
        num_classes=10,
        embed_dim=64,
        num_blocks=2,
        num_heads=8,
        dim_head=8,
        mlp_hidden_dim=128,
        dropout=0.1,
        device=device,
        deform_attn_activate=False
    )

    # Define Deformable Vision Transformer model
    dattn_model = nn.VisionTransformer(
        img_size=(32, 32),
        patch_size=4,
        in_channels=in_channels_dattn,
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

    print(f"in_channels_base = {in_channels_base}")
    print(f"in_channels_dattn = {in_channels_dattn}")
    print(f"len(self.attn.parmeters) = {sum(math.prod(each_w.shape) for each_w in base_model.transformer_blocks.modules[0].layer1.modules[0].parameters())}")
    print(f"len(self.dattn.parmeters) = {sum([math.prod(each_w.shape) for each_w in dattn_model.dattn.fn.modules[0].parameters()])}")
    print(f"len(self.base_model.parmeters) = {sum(math.prod(each_w.shape) for each_w in base_model.parameters())}")
    print(f"len(self.dattn_model.parmeters) = {sum(math.prod(each_w.shape) for each_w in dattn_model.parameters())}")

    # Train the baseline model
    train_acc_base, train_losses_base = train_cifar10(base_model, train_loader, n_epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001)

    # Train the deformable attention model
    train_acc_dattn, train_losses_dattn = train_cifar10(dattn_model, train_loader, n_epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001)

    # Evaluate the baseline model
    evaluate_cifar10(base_model, test_loader)

    # Evaluate the deformable attention model
    evaluate_cifar10(dattn_model, test_loader)

    # Plot
    plot_figures(train_losses_base, train_losses_dattn, train_acc_base, train_acc_dattn)