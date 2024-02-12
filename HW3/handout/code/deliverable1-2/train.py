import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from mytorch import MyConv2D, MyMaxPool2D


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):

        """
        My custom network
        [hint]
        * See the instruction PDF for details
        * Only allow to use MyConv2D and MyMaxPool2D
        * Set the bias argument to True
        """
        super().__init__()
        
        ## Define all the layers
        ## Use MyConv2D, MyMaxPool2D for the network
        # ----- TODO -----
        self.conv1 = MyConv2D(1, 3, 3, 1, 1)
        self.pool1 = MyMaxPool2D(kernel_size=2, stride=2)
        self.conv2 = MyConv2D(3, 6, 3, 1, 1)
        self.pool2 = MyMaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6*7*7, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        # ----- TODO -----
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy, total_loss 

def validate(model, device, validation_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print(f'Validation: Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy, total_loss



if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 10
    lr = 1e-4

    ## Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ----- TODO -----
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

    print(f"LOAD DATASET: TRAIN {len(trainset)} | TEST: {len(valset)}")

    ## Load my neural network
    # ----- TODO -----
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ## Define the criterion and optimizer
    # ----- TODO -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    ## Hint: you could separate the training and evaluation
    ## process into 2 different functions for each epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epoch): 
        # ----- TODO -----
        t_acc, t_loss = train(model, device, trainloader, optimizer, criterion)
        v_acc, v_loss = validate(model, device, valloader, criterion)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accuracies.append(t_acc)
        val_accuracies.append(v_acc)

    ## Plot the loss and accuracy curves
    # ----- TODO -----
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()


