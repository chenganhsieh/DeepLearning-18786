import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import MyResnet, init_weights_kaiming


def setup_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cam(net, inputs, labels, idx):

    """
    Calculate the CAM.

    [input]
    * net     : network
    * inputs  : input data
    * labels  : label data
    * idx     : the index of the chosen image in a minibatch, range: [0, batch_size-1]

    [output]
    * cam_img : CAM result
    * img     : raw image

    [hint]
    * Inputs and labels are in a minibatch form
    * You can choose one images from them for CAM by idx.
    """

    net.eval()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs, feat_map = net(inputs, return_embed=True)
        
        ## Find the class with highest probability
        ## Obtain the weight related to that class
        # ----- TODO -----
        _, predicted_class = outputs.max(1)
        class_idx = predicted_class[idx].item()

        ## Calculate the CAM
        ## Hint: you can choose one of the image (idx) from the batch for the following process
        # ----- TODO -----
        weight_softmax = net.fc.weight[class_idx].detach()
        cam = torch.matmul(weight_softmax, feat_map[idx].reshape(feat_map.size(1), -1))
        cam = cam.reshape(1,1,feat_map.size(2), feat_map.size(3)) 
        cam = torch.nn.functional.interpolate(cam, (inputs.size(2), inputs.size(3)), mode='bilinear')[0,0,:,:]
        
        ## Normalize CAM 
        ## Hint: Just minmax norm and rescale every value between [0-1]
        ## You will want to resize the CAM result for a better visualization
        ## e.g., the size of the raw image.
        # ----- TODO -----
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)  # Normalize between 0 and 1
        cam_img = cam.cpu().numpy()

        ## Denormalize raw images
        ## Hint: reverse the transform we did before
        ## Change the image data type into uint8 for visualization
        # ----- TODO -----
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = inputs[idx].permute(1, 2, 0).cpu().numpy() * std + mean  # Denormalize
        img = np.clip(img, 0, 1)

        return cam_img, img

def train(model, train_loader, device, criterion, optimizer,epoch):
    model.train()
    total_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch {epoch}: Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy, total_loss 
    
def evaluate(model, validation_loader, device, criterion, epoch):
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
    print(f'Validation: Loss {epoch}: {total_loss:.4f}, Accuracy: {accuracy:.2f}% \n')
    return accuracy, total_loss

def plot_heatmap(net, inputs, labels, idx):
    cam_img, img = cam(net, inputs, labels, idx)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    cam_image = axs[1].imshow(cam_img, cmap='jet')
    axs[1].set_title('CAM Result')
    axs[1].axis('off')

    axs[2].imshow(img)  
    axs[2].imshow(cam_img, cmap='jet', alpha=0.5)
    axs[2].set_title('Blending Result')
    axs[2].axis('off')

    # plt.tight_layout()
    # cbar = fig.colorbar(cam_image, ax=axs[1], fraction=0.046, pad=0.04)
    # cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
    cbar_ax = fig.add_axes([axs[2].get_position().x1 + 0.01, axs[2].get_position().y0, 0.02, axs[2].get_position().height])
    cbar = fig.colorbar(cam_image, cax=cbar_ax)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
    plt.show()
    plt.savefig(f"result_{idx}.png")



if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 120 # 113 85.42%
    lr = 1e-4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Set model
    ## Set the device to Cuda if needed
    ## Initialize all the parameters
    # ----- TODO -----
    net = MyResnet(in_channels=3, num_classes=10).to(DEVICE)
    net.load_state_dict(torch.load("model.pt"))
    # net.apply(init_weights_kaiming)


    ## Create the criterion and optimizer
    # ----- TODO -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    ## Load dataset
    normalize_param = dict(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    # ----- TODO -----
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"LOAD DATASET: TRAIN/VAL | {len(trainset)}/{len(valset)}")


    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    ## Hint: you could separate the training and evaluation 
    ## process into 2 different functions for each epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_acc = 0
    best_loss = float("inf")
    for epoch in range(num_epoch): 
        # ----- TODO -----
        t_acc, t_loss = train(net, trainloader, DEVICE, criterion, optimizer, epoch)
        v_acc, v_loss = evaluate(net, valloader, DEVICE, criterion, epoch)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accuracies.append(t_acc)
        val_accuracies.append(v_acc)
        if best_acc < v_acc:
            torch.save(net.state_dict(), "model.pt")
        best_loss = min(best_loss, v_loss)
        best_acc = max(best_acc, v_acc)
    print(f"Best ACC:{best_acc}, Best Loss:{best_loss}")
    print('Finished Training')

    ## Visualization
    ## Plot the loss and acc curves
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
    plt.savefig("deliverable3_5.png")


    ## Plot the CAM resuls as well as raw images
    ## Hint: You will want to resize the CAM result.
    # ----- TODO -----
    dataiter = iter(valloader)  # Assume trainloader is your DataLoader instance
    inputs, labels = next(dataiter)

    for i in range(60):
        plot_heatmap(net, inputs, labels, i)
    # plot_heatmap(net, inputs, labels, 10)
    # plot_heatmap(net, inputs, labels, 18)
    # plot_heatmap(net, inputs, labels, 25)
    


    



