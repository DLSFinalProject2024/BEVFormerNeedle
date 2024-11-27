# import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import ResNet9
from apps.simple_ml import train_cifar10, evaluate_cifar10
from needle.data import DataLoader, CIFAR10Dataset


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10Dataset(base_folder="./data/cifar-10-batches-py", train=True)
    test_dataset = CIFAR10Dataset(base_folder="./data/cifar-10-batches-py", train=False)

    # Create DataLoader instances
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
    # Define Vision Transformer model
    model = nn.VisionTransformer(
        img_size=(32, 32),
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        num_blocks=1,
        num_heads=8,
        dim_head=8,
        mlp_hidden_dim=128,
        dropout=0.1,
        device=device,
    )

    # Train the model
    train_cifar10(model, train_loader, n_epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001)

    # Evaluate the model
    evaluate_cifar10(model, test_loader)