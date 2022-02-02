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

model_dir = os.path.join('models', 'lenet-5')
now = datetime.now()
date_dir = os.path.join(model_dir, now.strftime('%m_%d_%Y'))
time_dir = os.path.join(date_dir, now.strftime('%H_%M_%S'))
weights_dir = os.path.join(time_dir, 'weights')
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)

model = PyLeNet5()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

train_set = MNIST('data/MNIST', train=True, download=True,
                  transform=T.Compose([
                      T.Resize((32, 32)),
                      T.ToTensor()]))

test_set = MNIST('data/MNIST', train=False, download=True,
                 transform=T.Compose([
                     T.Resize((32, 32)),
                     T.ToTensor()]))

train_batch_size = 128
test_batch_size = 128
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_batch_size)

# Initialize plots
axes = plt.figure().add_subplot(111)
axes.set_autoscale_on(True)
train_plot, = plt.plot([], [], 'r.-', label='Train loss')
test_plot, = plt.plot([], [], 'g.-', label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()


#########################
#       Training        #
#########################
train_epochs = []
test_epochs = []
train_losses = []
test_losses = []
for epoch in range(50):
    print(f'[~] Epoch {epoch}: ')

    train_loss = 0
    for images, target in tqdm(train_loader):
        optimizer.zero_grad()
        output = model.forward(images)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader) * train_batch_size
    writer.add_scalar('Loss/train', train_loss, epoch)
    train_epochs.append(epoch)
    train_losses.append(train_loss)

    if epoch % 1 == 0:
        torch.save(model.state_dict(), os.path.join(weights_dir, f'cp_{epoch}'))
        print(f' ~  Testing: ')
        test_loss = 0
        with torch.no_grad():
            for images, target in tqdm(test_loader):
                output = model.forward(images)
                loss = loss_function(output, target)

                test_loss += loss.item() / len(test_loader) * train_batch_size
        writer.add_scalar('Loss/test', test_loss, epoch)
        test_epochs.append(epoch)
        test_losses.append(test_loss)

    train_plot.set_data(train_epochs, train_losses)
    test_plot.set_data(test_epochs, test_losses)
    axes.relim()
    axes.autoscale_view(True, True, True)
    plt.tight_layout()
    plt.draw()
    plt.savefig(os.path.join(time_dir, 'losses.png'))

torch.save(model.state_dict(), os.path.join(weights_dir, 'final'))
