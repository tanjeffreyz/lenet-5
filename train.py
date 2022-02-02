"""Training grounds for Py-LeNet-5."""

import torch
import os
import torchvision.transforms as T
from torchvision.datasets.mnist import MNIST
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from models import PyLeNet5
from matplotlib import pyplot as plt


writer = SummaryWriter()

# Make directories
model_dir = os.path.join('models', 'lenet-5')
now = datetime.now()
date_dir = os.path.join(model_dir, now.strftime('%m_%d_%Y'))
time_dir = os.path.join(date_dir, now.strftime('%H_%M_%S'))
weights_dir = os.path.join(time_dir, 'weights')
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)

# Initialize globals
model = PyLeNet5()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

train_set = MNIST('data', train=True, download=True,
                  transform=T.Compose([
                      T.Resize((32, 32)),
                      T.ToTensor()]))

test_set = MNIST('data', train=False, download=True,
                 transform=T.Compose([
                     T.Resize((32, 32)),
                     T.ToTensor()]))

train_batch_size = 128
test_batch_size = 128
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_batch_size)

# Initialize plots
fig, axes = plt.subplots(2, 1)

losses = axes[0]
losses.set_xlabel('Epoch')
losses.set_ylabel('Loss')
losses.set_yscale('log')
losses.plot([], [], 'b.-', label='Train loss')
losses.plot([], [], 'g.-', label='Test loss')
losses.legend()

accuracies = axes[1]
accuracies.set_xlabel('Epoch')
accuracies.set_ylabel('Accuracy')
accuracies.plot([], [], 'b.-', label='Train accuracy')
accuracies.plot([], [], 'g.-', label='Test accuracy')
accuracies.legend()


#########################
#       Training        #
#########################
train_epochs, test_epochs = [], []
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
min_loss, min_loss_pt, prev_loss_ann = float('inf'), (0, 0), None
max_acc, max_acc_pt, prev_acc_ann = float('-inf'), (0, 0), None

for epoch in tqdm(range(25)):
    train_loss = 0
    accuracy = 0
    for images, targets in train_loader:
        optimizer.zero_grad()
        output = model.forward(images)
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_set) * train_batch_size
        accuracy += targets.eq(torch.argmax(output, 1)).sum().item() / len(train_set)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    train_epochs.append(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(accuracy)

    #########################
    #       Validation      #
    #########################
    if epoch % 1 == 0:
        torch.save(model.state_dict(), os.path.join(weights_dir, f'cp_{epoch}'))

        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, targets in test_loader:
                output = model.forward(images)
                loss = loss_function(output, targets)

                test_loss += loss.item() / len(test_set) * test_batch_size
                accuracy += targets.eq(torch.argmax(output, 1)).sum().item() / len(test_set)
        model.train()

        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        test_epochs.append(epoch)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        if test_loss < min_loss:
            min_loss = test_loss
            min_loss_pt = (epoch, test_loss)
        if accuracy > max_acc:
            max_acc = accuracy
            max_acc_pt = (epoch, accuracy)

    axes = axes.ravel()
    losses = axes[0]
    losses.plot(train_epochs, train_losses, 'b.-', label='Train loss')
    losses.plot(test_epochs, test_losses, 'g.-', label='Test loss')
    if prev_loss_ann:
        prev_loss_ann.remove()
    prev_loss_ann = losses.annotate(
        f'({min_loss_pt[0]}, {round(min_loss_pt[1], 3)})',
        xy=min_loss_pt, xycoords='data',
        xytext=(0, 50), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='green'),
        ha='center'
    )

    accuracies = axes[1]
    accuracies.plot(train_epochs, train_accuracies, 'b.-', label='Train accuracy')
    accuracies.plot(test_epochs, test_accuracies, 'g.-', label='Test accuracy')
    if prev_acc_ann:
        prev_acc_ann.remove()
    prev_acc_ann = accuracies.annotate(
        f'({max_acc_pt[0]}, {round(max_acc_pt[1], 3)})',
        xy=max_acc_pt, xycoords='data',
        xytext=(0, -50), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='green'),
        ha='center'
    )

    plt.tight_layout()
    plt.savefig(os.path.join(time_dir, 'losses.png'))

torch.save(model.state_dict(), os.path.join(weights_dir, 'final'))
