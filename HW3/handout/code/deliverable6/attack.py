#import packages
#feel free to import more if you need
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt



#evaluate the benign accuracy of a model
def test(model, x,y,batch_size):
    model.eval()
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    success=0
    loss=0
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=torch.tensor(x[start_index:end_index]).float()
        y_batch=torch.tensor(y[start_index:end_index]).long()
        output=model(x_batch)
        pred=torch.argmax(output,dim=1)
        loss+=F.cross_entropy(output,y_batch).item()
        success+=(pred==y_batch).sum().item()
    print ("accuracy: "+str(success/total))


def evaluate_attack_success_rate(model, adv_examples, true_labels, target_class=None, attack_class=None):
    model.eval()
    true_labels = torch.tensor(true_labels).long()

    if target_class and attack_class:
        original_class_mask = (true_labels == attack_class)
        
        adv_examples = adv_examples[original_class_mask]
        true_labels = true_labels[original_class_mask]
        target_class = torch.full((true_labels.shape[0],), target_class)
        
        if len(adv_examples) == 0:
            return 0.0


    with torch.no_grad():
        outputs = model(adv_examples)
    
    _, predicted_classes = torch.max(outputs, 1)
    
    if target_class is None:
        successful_attacks = predicted_classes != true_labels
    else:
        successful_attacks = predicted_classes == target_class
    
    success_rate = successful_attacks.float().mean().item()
    
    return success_rate

def plot_adversarial_examples(original_images, adversarial_images, n_examples=5, epsilon=2, model_name=None, att_name=None):
    plt.figure(figsize=(10, 2 * n_examples))
    adv_img = adversarial_images.clone().detach()
    
    for i in range(n_examples):
        plt.subplot(n_examples, 2, 2*i + 1)
        plt.imshow(original_images[i].squeeze(), cmap='gray' if original_images[i].shape[0] == 1 else None)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(n_examples, 2, 2*i + 2)
        plt.imshow(adv_img[i].squeeze(), cmap='gray' if adv_img[i].shape[0] == 1 else None)
        plt.title(f"Adversarial ε={epsilon}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"result_{model_name}_{att_name}_{epsilon}.png")



#define model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load MNIST
dataset_train = datasets.MNIST('../data', train=True, download=True)
dataset_test = datasets.MNIST('../data', train=False, download=True)

# reshape MNIST
x_train=dataset_train.data.numpy()
y_train=dataset_train.targets.numpy()
x_test=dataset_test.data.numpy()
y_test=dataset_test.targets.numpy()
y_test_tensor = torch.tensor(y_test).long()
x_train=np.reshape(x_train,(60000,28,28,1))
x_test=np.reshape(x_test,(10000,28,28,1))
x_train=np.swapaxes(x_train, 1, 3)
x_test=np.swapaxes(x_test, 1, 3)
x_test_tensor = torch.tensor(x_test).float()
x_test_tensor = x_test_tensor / 255

#REMINDER: the range of inputs is different from what we used in the recitation
print (x_test.min(),x_test.max())


modelA=Net()
# modelA.to(DEVICE)
modelA.load_state_dict(torch.load("modelA.zip"))
test(modelA,x_test,y_test,512)
modelB=Net()
# modelB.to(DEVICE)
modelB.load_state_dict(torch.load("modelB.zip"))
test(modelB,x_test,y_test,512)


#untargeted attack
#you may add parameters as you wish
def untargeted_attack(model, images, labels, epsilon=2, iters=20):
    model.eval()
    epsilon = epsilon / 255.

    perturbed_images = images.clone().detach().requires_grad_(True)
    for _ in range(iters):
        outputs = model(perturbed_images)

        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        sign_data_grad = perturbed_images.grad.data.sign()
        with torch.no_grad():
            perturbed_images = perturbed_images + epsilon * sign_data_grad
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            perturbed_images = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon) + images
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            perturbed_images = perturbed_images.detach().requires_grad_(True)
            perturbed_images = torch.round(perturbed_images*255)/255
    perturbed_images.requires_grad_(False)
    return perturbed_images


#targeted attack
#you may add parameters as you wish
def targeted_attack(model, images, labels, target_class, epsilon=2, iters=100):
    model.eval()
    epsilon = epsilon / 255.
    # Make a copy of the images to avoid modifying the original ones
    perturbed_images = images.clone().detach().requires_grad_(True)
    target_labels = torch.full_like(labels, target_class)  # Create a tensor of the target class labels

    for _ in range(iters):
        outputs = model(perturbed_images)

        loss = F.cross_entropy(outputs, target_labels)
        model.zero_grad()
        loss.backward()

        sign_data_grad = perturbed_images.grad.data.sign()
        with torch.no_grad():
            perturbed_images = perturbed_images - epsilon * sign_data_grad
            perturbed_images = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon) + images
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            perturbed_images = perturbed_images.detach().requires_grad_(True)
            perturbed_images = torch.round(perturbed_images*255)/255
    
    perturbed_images.requires_grad_(False)
    return perturbed_images
         
#improved targeted attack 
#you may add parameters as you wish
def targeted_attack_improved(model, images, labels, target_class, epsilon=0.03, iters=50, decay_factor=1.0):
    model.eval()
    epsilon = epsilon / 255.
    perturbed_images = images.clone().detach().requires_grad_(True)
    target_labels = torch.full_like(labels, target_class)  # Create a tensor of the target class labels
    momentum = torch.zeros_like(images)

    for _ in range(iters):
        outputs = model(perturbed_images)

        loss = F.cross_entropy(outputs, target_labels)
        model.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        momentum = decay_factor * momentum + grad / grad.norm(p=1)
        perturbed_images = perturbed_images - epsilon * momentum.sign()

        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon) + images
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        perturbed_images = perturbed_images.detach().requires_grad_(True)
    
    # perturbed_images = torch.round(perturbed_images*255)/255

    return perturbed_images

#evaluate performance of attacks
	#TODO

models = [modelA, modelB]
model_names = ['Model A', 'Model B']
attacks = [untargeted_attack, targeted_attack, targeted_attack_improved]
attack_names = ['Untargeted Attack', 'Targeted Attack', 'Improved Targeted Attack']
epsilons = [2, 4, 8, 16]

# Placeholder for success rates
success_rates = np.zeros((len(models), len(attacks), len(epsilons)))

for i, model in enumerate(models):
    print(f"Model:{model_names[i]}")
    for j, attack in enumerate(attacks):
        print(f"Attack:{attack_names[j]}")
        for k, epsilon in enumerate(epsilons):
            print(f"Epsilon: {epsilon}")
            if j > 0:
                adv_examples = attack(model=model, images=x_test_tensor, labels=y_test_tensor, target_class= 8, epsilon=epsilon)
                success_rate = evaluate_attack_success_rate(model, adv_examples, y_test_tensor, target_class=8, attack_class=1)
            else:
                adv_examples = attack(model=model, images=x_test_tensor, labels=y_test_tensor, epsilon=epsilon)
                success_rate = evaluate_attack_success_rate(model, adv_examples, y_test_tensor)

            
            original_images = x_test[:3]
            adversarial_images = adv_examples[:3]
            plot_adversarial_examples(original_images, adversarial_images, 3, epsilon, model_names[i], attack_names[j])
            
            test(model, adv_examples, y_test, 512)
            success_rates[i, j, k] =  
            print(f"Attack Success Rate:{success_rate}")
            
    print("\n")

# Plotting
for i in range(len(models)):
    plt.figure(figsize=(10, 6))
    for j in range(len(attacks)):
        plt.plot(epsilons, success_rates[i, j, :], marker='o', label=attack_names[j])
    plt.title(f'Success Rate vs Epsilon for {model_names[i]}')
    plt.xlabel('Epsilon')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_names[i]}.png")