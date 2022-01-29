"""Training grounds for Py-LeNet-5."""

import torch
import time
import os
import torchvision.transforms as T
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import PyLeNet5
from matplotlib import pyplot as plt


model_dir = 'models/lenet-5'
weights_dir = os.path.join(model_dir, 'weights')
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)

model = PyLeNet5()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

train_set = MNIST('data/MNIST',
                  train=True,
                  download=True,
                  transform=T.Compose([
                      T.Resize((32, 32)),
                      T.ToTensor()]))

test_set = MNIST('data/MNIST',
                 train=False,
                 download=True,
                 transform=T.Compose([
                     T.Resize((32, 32)),
                     T.ToTensor()]))

train_batch_size = 128
test_batch_size = 128
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_batch_size)

# Initialize plots
fig, axs = plt.subplots(2, 1)
losses = axs[0]
losses.set_xlabel('Epoch')
losses.set_ylabel('Loss')


#########################
#       Training        #
#########################
train_epochs = []
test_epochs = []
train_losses = []
test_losses = []
for epoch in range(1_000):
    print(f'[~] Epoch {epoch}: ')
    train_loss = 0
    for image, target in tqdm(train_loader):
        optimizer.zero_grad()
        output = model.forward(image)
        loss = loss_function(output, target)
        loss.backward()

        train_loss += loss.item() / len(train_loader) * train_batch_size

    train_epochs.append(epoch)
    train_losses.append(train_loss)

    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), os.path.join(weights_dir, f'cp_{epoch}'))

        print(f' ~  Testing: ')
        test_loss = 0
        with torch.no_grad():
            for image, target in tqdm(test_loader):
                output = model.forward(image)
                loss = loss_function(output, target)

                test_loss += loss.item() / len(test_loader) * train_batch_size

        test_epochs.append(epoch)
        test_losses.append(test_loss)

    axs = axs.ravel()
    losses = axs[0]
    losses.plot(train_epochs, train_losses, 'r.-', label='Train loss')
    losses.plot(test_epochs, test_losses, 'g.-', label='Test loss')
    if epoch == 0:
        losses.legend()
    plt.draw()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'losses.png'))

torch.save(model.state_dict(), os.path.join(weights_dir, 'final'))
