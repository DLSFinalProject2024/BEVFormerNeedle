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
    ### BEGIN YOUR SOLUTION
    two_layer_linear = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim), 
                                     norm(dim=hidden_dim), 
                                     nn.ReLU(), 
                                     nn.Dropout(p=drop_prob),
                                     nn.Linear(in_features=hidden_dim, out_features=dim),
                                     norm(dim=dim)) 

    out_module = nn.Sequential(nn.Residual(two_layer_linear),
                               nn.ReLU())

    return out_module
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    pre_module = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim),
                               nn.ReLU())
    residual_block = []
    for i in range(num_blocks):
        residual_block.append(ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob))

    return nn.Sequential(pre_module,
                         *residual_block,
                         nn.Linear(in_features=hidden_dim, out_features=num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    avg_loss = []
    avg_err  = []
    if opt is not None:
        # Training Mode
        model.train()
        for i, batch in enumerate(dataloader):
            opt.reset_grad()
            batch_x, batch_y = batch[0], batch[1]
            batch_x = ndl.nn.Flatten()(batch_x)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            loss.backward()
            opt.step()
            avg_loss.append(np.float32(loss.numpy()))
            avg_err.append(np.float32(np.sum(batch_y.numpy() != out.numpy().argmax(axis=1))))
    else:
        # Testing Mode
        model.eval()
        for i, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            batch_x = ndl.nn.Flatten()(batch_x)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            avg_loss.append(np.float32(loss.numpy()))
            avg_err.append(np.float32(np.sum(batch_y.numpy() != out.numpy().argmax(axis=1))))

    avg_loss_val = np.mean(avg_loss)
    avg_err_val  = np.sum(avg_err)/len(dataloader.dataset)
    return avg_err_val, avg_loss_val

    ### END YOUR SOLUTION


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
    ### BEGIN YOUR SOLUTION
    # Setting
    train_image_file = f"{data_dir}/train-images-idx3-ubyte.gz"
    train_label_file = f"{data_dir}/train-labels-idx1-ubyte.gz"
    test_image_file = f"{data_dir}/t10k-images-idx3-ubyte.gz"
    test_label_file = f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    train_dataset    = ndl.data.MNISTDataset(train_image_file, train_label_file)
    test_dataset     = ndl.data.MNISTDataset(test_image_file, test_label_file)
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = ndl.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    for loop_i in range(epochs):
        err_train, loss_train = epoch(dataloader=train_dataloader, model=model, opt=opt)

    # Testing
    err_test, loss_test = epoch(dataloader=test_dataloader, model=model, opt=None)

    return err_train, loss_train, err_test, loss_test
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
