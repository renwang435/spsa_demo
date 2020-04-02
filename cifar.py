from __future__ import print_function

import argparse
import os
import sys
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.theta = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        output = self.theta(x)

        return output

# TODO
'''
We should change pass in:
a) loss (inherit from _Loss in torch.nn.modules, loss.py)
b) optimizer (inherit from Optimizer in torch.optim, optimizer.py)
'''
def train(args, model, device, train_loader, loss, optimizer, epoch):
    # Create model name
    save_name = 'mnist_{}_{}_{}_{}'.format(
            args.random_seed, args.lr, args.batch_size, args.epochs
        )
    ckpt_dir = './ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = np.inf
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss_out = loss(output, target)
        loss_out.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_out.item()))

        is_best = loss_out.item() < best_loss
        best_loss = min(best_loss, loss_out.item())
        if is_best:
                save_checkpoint(ckpt_dir, save_name,
                    {'epoch': epoch,
                     'model_state': model.state_dict(),
                     'optim_state': optimizer.state_dict(),
                     'best_loss': best_loss
                     }, is_best
                )


def test(args, model, device, test_loader, loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, loss, optimizer, epoch)
        test(args, model, device, test_loader, loss)


def save_checkpoint(ckpt_dir, model_name, state, is_best):
    """
    Save a copy of the model so that it can be loaded at a future
    date. This function is used when the model is being evaluated
    on the test data.

    If this model has reached the best validation accuracy thus
    far, a seperate file with the suffix `best` is created.
    """
    print("[*] Saving model to {}".format(ckpt_dir))

    root = model_name

    filename = root + '_ckpt.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

    if is_best:
        filename = root + '_model_best.pth.tar'
        shutil.copyfile(
            ckpt_path, os.path.join(ckpt_dir, filename)
        )

def load_checkpoint(model_name, ckpt_dir, optimizer, best=False):
    """
    Load the best copy of a model. This is useful for 2 cases:

    - Resuming training with the most recent model checkpoint.
    - Loading the best validation model to evaluate on the test data.

    Params
    ------
    - best: if set to True, loads the best model. Use this if you want
        to evaluate your model on the test data. Else, set to False in
        which case the most recent version of the checkpoint is used.
    """
    print("[*] Loading model from {}".format(ckpt_dir))

    ds_model = model_name + '_ckpt.pth.tar'
    if best:
        ds_model = model_name + '_model_best.pth.tar'
    ds_ckpt_path = os.path.join(ckpt_dir, ds_model)
    ds_ckpt = torch.load(ds_ckpt_path)

    # load variables from checkpoint
    model = Net()
    model.load_state_dict(ds_ckpt['model_state'])
    optimizer.load_state_dict(ds_ckpt['optim_state'])

    print("Successfully loaded model...")

    if best:
        print(
            "[*] Loaded {} checkpoint @ epoch {} "
            "with best valid acc of {:.3f}".format(
                ds_model, ds_ckpt['epoch'], ds_ckpt['best_val_loss'])
        )
    else:
        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ds_model, ds_ckpt['epoch'])
        )

if __name__ == '__main__':
    main()
