import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    block = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(block), nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    blocks = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        blocks.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    
    blocks.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*blocks)


def epoch(dataloader: ndl.data.DataLoader, model: nn.Module, opt: ndl.optim.Optimizer=None):
    np.random.seed(4)
    
    model.eval() if opt is None else model.train()
    loss_fn = nn.SoftmaxLoss()
    sum_err = 0.0
    sum_loss = 0.0

    for i, batch in enumerate(dataloader):
        X, y = batch
        X = X.reshape((X.shape[0], -1)) # Flatten: (B, 28, 28, 1) -> (B, 784)
        batch_size = y.shape[0]

        if model.training:
            opt.reset_grad()

        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()

        if model.training:
            loss.backward()
            opt.step()

        sum_err += (logits.numpy().argmax(-1) != y.numpy()).sum().item()
        sum_loss += loss.numpy().item() * batch_size

    return sum_err / len(dataloader.dataset), sum_loss / len(dataloader.dataset)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)

    # Train Data
    train_img_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_lbl_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    train_dataset = ndl.data.MNISTDataset(train_img_path, train_lbl_path)
    train_loader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Test Data
    test_img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_lbl_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(test_img_path, test_lbl_path)
    test_loader = ndl.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model & Optimizer
    model = MLPResNet(28 * 28, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)

    # Evaluation
    test_err, test_loss = epoch(test_loader, model)
    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
