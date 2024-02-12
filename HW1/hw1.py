# Problem 1.
# Your code goes here
import torch
import torch.nn as nn
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# define the data pre-processing
# convert the input to the range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset
# this command requires Internet to download the dataset
mnist = datasets.MNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transform)
mnist_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)

# Define your MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        # Your code goes here
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        # Your code goes here
        return self.fc2(self.activation(self.fc1(x)))

# Your code goes here
hidden_dim = 64
criterion = nn.CrossEntropyLoss()


# Filter for digits 0 and 1
train_data = [data for data in mnist if data[1] < 2]
# Your code goes here
test_data = [data for data in mnist_test if data[1] < 2]

# Split training data into training and validation sets
# Your code goes here
split_index = int(0.8 * len(train_data))
train_set = train_data[:split_index]
val_set = train_data[split_index:]

# Define DataLoaders to access data in batches
batch_sizes = [2,16,128,1024]

for batch_size in batch_sizes:
  start = time.time()
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  # Your code goes here
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
  model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 10

  train_correct = train_total = 0
  for epoch in range(num_epochs):
      for data, target in train_loader:
          # free the gradient from the previous batch
          optimizer.zero_grad()
          # reshape the image into a vector
          data = data.view(data.size(0), -1)
          # model forward
          output = model(data)
          # compute the loss
          loss = criterion(output, target)
          # model backward
          loss.backward()
          # update the model paramters
          optimizer.step()

          if epoch == num_epochs - 1:
            output = model(data)
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += data.size(0)


  val_loss = val_count = 0
  val_correct = val_total = 0
  for data, target in val_loader:
      data = data.view(data.size(0), -1)
      output = model(data)
      val_loss += criterion(output, target).item()
      val_count += 1
      pred = output.argmax(dim=1)
      val_correct += (pred == target).sum().item()
      val_total += data.size(0)

  test_loss = test_count = 0
  test_correct = test_total = 0

  for data, target in test_loader:
      data = data.view(data.size(0), -1)
      output = model(data)
      test_loss += criterion(output, target).item()
      test_count += 1
      pred = output.argmax(dim=1)
      test_correct += (pred == target).sum().item()
      test_total += data.size(0)

  train_acc = 100. * train_correct / train_total
  val_loss = val_loss / val_count
  val_acc = 100. * val_correct / val_total
  test_loss = test_loss / test_count
  test_acc = 100. * test_correct / test_total
  end = time.time()
  print(f"Batch size:{batch_size}, Execute time:{end-start}, Train acc:{train_acc}, Val acc:{val_acc}, Test acc:{test_acc}")

# ==========================================================================================================================================
# Problem 2.
# Your code goes here
import torch
import torch.nn as nn
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
torch.manual_seed(0)

# define the data pre-processing
# convert the input to the range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset
# this command requires Internet to download the dataset
mnist = datasets.MNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transform)
mnist_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)
# Split training data into training and validation sets
# Your code goes here
# Filter for digits 0 and 1
train_data = [data for data in mnist]
split_index = int(0.8 * len(train_data))
train_set = train_data[:split_index]
val_set = train_data[split_index:]
test_set = mnist_test


train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)




# Define your MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        # Your code goes here
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        # Your code goes here
        return self.fc2(self.activation(self.fc1(x)))

# Your code goes here
hidden_dims = [32, 64, 128, 256]
criterion = nn.CrossEntropyLoss()

for hidden_dim in hidden_dims:
  start = time.time()
  model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=10)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 10

  train_correct = train_total = 0
  for epoch in range(num_epochs):
      for data, target in train_loader:
          # free the gradient from the previous batch
          optimizer.zero_grad()
          # reshape the image into a vector
          data = data.view(data.size(0), -1)
          # model forward
          output = model(data)
          # compute the loss
          loss = criterion(output, target)
          # model backward
          loss.backward()
          # update the model paramters
          optimizer.step()

          if epoch == num_epochs - 1:
            output = model(data)
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += data.size(0)


  val_loss = val_count = 0
  val_correct = val_total = 0
  for data, target in val_loader:
      data = data.view(data.size(0), -1)
      output = model(data)
      val_loss += criterion(output, target).item()
      val_count += 1
      pred = output.argmax(dim=1)
      val_correct += (pred == target).sum().item()
      val_total += data.size(0)

  test_loss = test_count = 0
  test_correct = test_total = 0

  for data, target in test_loader:
      data = data.view(data.size(0), -1)
      output = model(data)
      test_loss += criterion(output, target).item()
      test_count += 1
      pred = output.argmax(dim=1)
      test_correct += (pred == target).sum().item()
      test_total += data.size(0)

  train_acc = 100. * train_correct / train_total
  val_loss = val_loss / val_count
  val_acc = 100. * val_correct / val_total
  test_loss = test_loss / test_count
  test_acc = 100. * test_correct / test_total
  end = time.time()
  print(f"Hidden size:{hidden_dim}, Execute time:{end-start}, Train acc:{train_acc}, Val acc:{val_acc}, Test acc:{test_acc}")

# ==========================================================================================================================================
# Problem 3.
# Your code goes here
import torch
import torch.nn as nn
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
torch.manual_seed(0)

def calculate_f1_score(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# define the data pre-processing
# convert the input to the range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset
# this command requires Internet to download the dataset
mnist = datasets.MNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transform)
mnist_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)
# Split training data into training and validation sets
# Your code goes here
train_0 = [data for data in mnist if data[1] == 0]
train_1 = [data for data in mnist if data[1] == 1]
train_1 = train_1[:len(train_1) // 100]
train_data = train_0 + train_1
# Your code goes here
test_data = [data for data in mnist_test if data[1] < 2]


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)



# Define your MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

criterion = nn.CrossEntropyLoss()
hidden_dim = 128

model = SimpleMLP(in_dim=28 * 28,
                hidden_dim=hidden_dim,
                out_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

train_correct = train_total = 0
TP, FP, FN = 0, 0, 0
for epoch in range(num_epochs):
    for data, target in train_loader:
        # free the gradient from the previous batch
        optimizer.zero_grad()
        # reshape the image into a vector
        data = data.view(data.size(0), -1)
        # model forward
        output = model(data)
        # compute the loss
        loss = criterion(output, target)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()

        if epoch == num_epochs - 1:
          output = model(data)
          pred = output.argmax(dim=1)
          train_correct += (pred == target).sum().item()
          train_total += data.size(0)
          TP += ((pred == 1) & (target == 1)).sum().item()
          FP += ((pred == 1) & (target == 0)).sum().item()
          FN += ((pred == 0) & (target == 1)).sum().item()
train_f1_score = calculate_f1_score(TP, FP, FN)
test_loss = test_count = 0
test_correct = test_total = 0

TP, FP, FN = 0, 0, 0

for data, target in test_loader:
    data = data.view(data.size(0), -1)
    output = model(data)
    test_loss += criterion(output, target).item()
    test_count += 1
    pred = output.argmax(dim=1)
    test_correct += (pred == target).sum().item()
    test_total += data.size(0)
    TP += ((pred == 1) & (target == 1)).sum().item()
    FP += ((pred == 1) & (target == 0)).sum().item()
    FN += ((pred == 0) & (target == 1)).sum().item()

train_acc = 100. * train_correct / train_total
test_loss = test_loss / test_count
test_acc = 100. * test_correct / test_total
f1_score = calculate_f1_score(TP, FP, FN)
print(f"Train acc:{train_acc}, Train F1 score: {train_f1_score}, Test acc:{test_acc}, F1 score:{f1_score}")

from torch.utils.data import WeightedRandomSampler

class_weights = torch.tensor([1.0, 100.0])
# if torch.cuda.is_available():
#     class_weights = class_weights.cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

train_0 = [data for data in mnist if data[1] == 0]
train_1 = [data for data in mnist if data[1] == 1]
train_1 = train_1[:len(train_1) // 100]
train_data = train_0 + train_1
test_data = [data for data in mnist_test if data[1] < 2]

# Calculate weights for each sample
sample_weights = [100 if label == 1 else 1 for data, label in train_data]  # Oversample class "1"
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_data, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

hidden_dim = 128

model = SimpleMLP(in_dim=28 * 28,
                hidden_dim=hidden_dim,
                out_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

train_correct = train_total = 0
TP, FP, FN = 0, 0, 0
for epoch in range(num_epochs):
    for data, target in train_loader:
        # free the gradient from the previous batch
        optimizer.zero_grad()
        # reshape the image into a vector
        data = data.view(data.size(0), -1)
        # model forward
        output = model(data)
        # compute the loss
        loss = criterion(output, target)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()

        if epoch == num_epochs - 1:
          output = model(data)
          pred = output.argmax(dim=1)
          train_correct += (pred == target).sum().item()
          train_total += data.size(0)
          TP += ((pred == 1) & (target == 1)).sum().item()
          FP += ((pred == 1) & (target == 0)).sum().item()
          FN += ((pred == 0) & (target == 1)).sum().item()
train_f1_score = calculate_f1_score(TP, FP, FN)
test_loss = test_count = 0
test_correct = test_total = 0

TP, FP, FN = 0, 0, 0

for data, target in test_loader:
    data = data.view(data.size(0), -1)
    output = model(data)
    test_loss += criterion(output, target).item()
    test_count += 1
    pred = output.argmax(dim=1)
    test_correct += (pred == target).sum().item()
    test_total += data.size(0)
    TP += ((pred == 1) & (target == 1)).sum().item()
    FP += ((pred == 1) & (target == 0)).sum().item()
    FN += ((pred == 0) & (target == 1)).sum().item()

train_acc = 100. * train_correct / train_total
test_loss = test_loss / test_count
test_acc = 100. * test_correct / test_total
f1_score = calculate_f1_score(TP, FP, FN)
print(f"Train acc:{train_acc}, Train F1 score: {train_f1_score}, Test acc:{test_acc}, F1 score:{f1_score}")

# ==========================================================================================================================================
# Problem 3. bonus
# Your code goes here
import torch
import torch.nn as nn
import time
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
torch.manual_seed(0)
random.seed(0)

def calculate_f1_score(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# define the data pre-processing
# convert the input to the range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset
# this command requires Internet to download the dataset
mnist = datasets.MNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transform)
mnist_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)
train_0 = [data for data in mnist if data[1] == 0]
train_1 = [data for data in mnist if data[1] == 1]
random.shuffle(train_1)

# Define your MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class_weights = torch.tensor([1.0, 50.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
hidden_dim = 128

model = SimpleMLP(in_dim=28 * 28,
                hidden_dim=hidden_dim,
                out_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
num_more_epochs = 1

for N in [500,1000,1500,2000,2500,3000]:
  # Split training data into training and validation sets
  # Your code goes here
  train_small_1 = train_1[:len(train_1) // N]

  transform = transforms.Compose([
      transforms.RandomRotation(30),
      transforms.RandomHorizontalFlip(),
  ])
  train_1_augmented = [(transform(x[0]), x[1]) for x in train_small_1]

  train_data = train_0 + train_small_1
  # Your code goes here
  test_data = [data for data in mnist_test if data[1] < 2]
  sample_weights = [100 if label == 1 else 1 for data, label in train_data]  # Oversample class "1"
  sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
  train_loader = DataLoader(train_data, batch_size=16, sampler=sampler)
  train_label1_loader = DataLoader(train_1_augmented, batch_size=16, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


  train_correct = train_total = 0
  TP, FP, FN = 0, 0, 0
  for epoch in range(num_epochs):
      for data, target in train_loader:
          # free the gradient from the previous batch
          optimizer.zero_grad()
          # reshape the image into a vector
          data = data.view(data.size(0), -1)
          # model forward
          output = model(data)
          # compute the loss
          loss = criterion(output, target)
          # model backward
          loss.backward()
          # update the model paramters
          optimizer.step()

          # if epoch == num_epochs - 1:
          #   output = model(data)
          #   pred = output.argmax(dim=1)
          #   train_correct += (pred == target).sum().item()
          #   train_total += data.size(0)
          #   TP += ((pred == 1) & (target == 1)).sum().item()
          #   FP += ((pred == 1) & (target == 0)).sum().item()
          #   FN += ((pred == 0) & (target == 1)).sum().item()
  for epoch in range(num_more_epochs):
      for data, target in train_label1_loader:
          # free the gradient from the previous batch
          optimizer.zero_grad()
          # reshape the image into a vector
          data = data.view(data.size(0), -1)
          # model forward
          output = model(data)
          # compute the loss
          loss = criterion(output, target)
          # model backward
          loss.backward()
          # update the model paramters
          optimizer.step()

          if epoch == num_more_epochs - 1:
            output = model(data)
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += data.size(0)
            TP += ((pred == 1) & (target == 1)).sum().item()
            FP += ((pred == 1) & (target == 0)).sum().item()
            FN += ((pred == 0) & (target == 1)).sum().item()
  train_f1_score = calculate_f1_score(TP, FP, FN)
  test_loss = test_count = 0
  test_correct = test_total = 0

  TP, FP, FN = 0, 0, 0

  for data, target in test_loader:
      data = data.view(data.size(0), -1)
      output = model(data)
      test_loss += criterion(output, target).item()
      test_count += 1
      pred = output.argmax(dim=1)
      test_correct += (pred == target).sum().item()
      test_total += data.size(0)
      TP += ((pred == 1) & (target == 1)).sum().item()
      FP += ((pred == 1) & (target == 0)).sum().item()
      FN += ((pred == 0) & (target == 1)).sum().item()

  train_acc = 100. * train_correct / train_total
  test_loss = test_loss / test_count
  test_acc = 100. * test_correct / test_total
  f1_score = calculate_f1_score(TP, FP, FN)
  print(f"N:{N}, Train acc:{train_acc}, Train F1 score: {train_f1_score}, Test acc:{test_acc}, F1 score:{f1_score}")

# ==========================================================================================================================================
# Problem 4.
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
random.seed(0)
torch.manual_seed(0)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

# Set the size of the input and the hidden layer
input_size = 784
hidden_size = 50
output_size = input_size  # Output size is the same as input for reconstruction
# define the data pre-processing
# convert the input to the range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset
# this command requires Internet to download the dataset
mnist = datasets.MNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transform)
mnist_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)
# Split training data into training and validation sets
train_data = [data for data in mnist]
split_index = int(0.8 * len(train_data))
train_set = train_data[:split_index]
val_set = train_data[split_index:]
test_set = mnist_test

# Data loaders
batch_size = 64
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
# Create the model
model = SimpleMLP(in_dim=input_size,
                hidden_dim=hidden_size,
                out_dim=output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



num_epochs = 40
for epoch in range(num_epochs):
    for data, _ in train_loader:
        # free the gradient from the previous batch
        optimizer.zero_grad()
        # reshape the image into a vector
        data = data.view(data.size(0), -1)
        # model forward
        output = model(data)
        # compute the loss
        loss = criterion(output, data)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()
    # Validation
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            val_loss += criterion(output, data).item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Test
test_loss = 0
with torch.no_grad():
    for data, _ in test_loader:
        data = data.view(data.size(0), -1)
        output = model(data)
        test_loss += criterion(output, data).item()
print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# Select 5 images for each digit
images_per_digit = 5
digit_images = {i: [] for i in range(10)}

with torch.no_grad():
    for data, target in test_loader:
        for i in range(data.size(0)):
            if len(digit_images[target[i].item()]) < images_per_digit:
                digit_images[target[i].item()].append(data[i])

fig, axs = plt.subplots(10, images_per_digit * 2, figsize=(15, 20))

for digit, imgs in digit_images.items():
    for i, img in enumerate(imgs):
        # Original Image
        img_original = img.view(-1)
        axs[digit, 2*i].imshow(img_original.reshape(28, 28), cmap='gray')
        axs[digit, 2*i].set_title(f'Original {digit}')
        axs[digit, 2*i].axis('off')

        # Reconstructed Image
        img_reconstructed = model(img_original).view(28, 28)
        axs[digit, 2*i+1].imshow(img_reconstructed.detach().numpy(), cmap='gray')
        axs[digit, 2*i+1].set_title(f'Reconstructed {digit}')
        axs[digit, 2*i+1].axis('off')

plt.show()