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
    true_labels = true_labels.clone().detach().long()
    
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
x_test_tensor = x_test_tensor

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
def untargeted_attack(model, images, labels, epsilon=2, iters=20, alpha = 1.0):
    model.eval()

    perturbed_images = images.detach().clone()
    for _ in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)

        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        sign_data_grad = perturbed_images.grad.sign()
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * sign_data_grad
            perturbed_images = torch.clamp(perturbed_images, 0, 255)
            perturbed_images = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon) + images
            perturbed_images = torch.clamp(perturbed_images, 0, 255).detach().clone()

    return perturbed_images


#targeted attack
#you may add parameters as you wish
def targeted_attack(model, images, labels, target_class, epsilon=2, iters=100, alpha=1.0):
    model.eval()
    
    mask = labels == 1
    images_to_attack = images[mask]
    labels_to_attack = labels[mask]
    perturbed_images = images_to_attack.detach().clone()
    target_labels = torch.full_like(labels_to_attack, target_class)
    for _ in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, target_labels)
        loss.backward()

        sign_data_grad = perturbed_images.grad.data
        with torch.no_grad():
            perturbed_images = perturbed_images - alpha * sign_data_grad.sign()
            perturbed_images = torch.clamp(perturbed_images, min=images_to_attack-epsilon, max=images_to_attack+epsilon)
            perturbed_images = torch.clamp(perturbed_images, 0, 255)
            perturbed_images = perturbed_images.detach().clone()
    
    perturbed_images.requires_grad = False
    return_images = images.detach().clone()
    return_images[mask] = perturbed_images
    return return_images


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 28 - 2*padding // stride + 1
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 784),
            nn.Tanh(),
            nn.Linear(784, 784),
        )
    def forward(self, x):
        return self.layers(x)
    
#improved targeted attack 
#you may add parameters as you wish
def targeted_attack_improved(model, images, labels, target_class, epsilons=[1,2,4,8,16,32], iters=100, alpha=1.0):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    mask_1 = labels == 1
    images_to_attack = images[mask_1].detach().clone()
    labels_to_attack = labels[mask_1].detach().clone()
    perturbed_images = images_to_attack.detach().clone()
    target_labels = torch.full_like(labels_to_attack, target_class) 
    
    batches=np.ceil(len(images_to_attack)/16).astype(int)
    success=0
    total_loss=0
    net = Autoencoder()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epsilon_large = 16
    for iter in range(400):
        total_loss = 0
        for i in range(batches):
            net.zero_grad()
            start_index=i*16
            end_index=np.minimum((i+1)*16,len(images_to_attack))
            x_batch=torch.tensor(images_to_attack[start_index:end_index].detach().clone()).float()
            y_batch=torch.tensor(target_labels[start_index:end_index].detach().clone()).long()
            perturbed_images = net(x_batch/255)
            perturbed_images = perturbed_images * 255
            perturbed_images = torch.clamp(perturbed_images, min=x_batch.detach().clone()-epsilon_large, max=x_batch.detach().clone()+epsilon_large)
            perturbed_images = torch.clamp(perturbed_images, 0, 255)
            output = model(perturbed_images)
            
            loss = F.cross_entropy(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {iter}: {total_loss / batches}")
    perturbed_images = images_to_attack.detach().clone()
    # perturbed_images = images_to_attack.detach().clone().reshape(images_to_attack.shape[0], -1)
    perturbed_images = net(perturbed_images/255)
    perturbed_images = perturbed_images * 255
    # perturbed_images = perturbed_images.reshape(perturbed_images.shape[0], 1,28, 28)
    perturbed_images = torch.clamp(perturbed_images, min=images_to_attack-epsilon_large, max=images_to_attack+epsilon_large)
    perturbed_images = torch.clamp(perturbed_images, 0, 255)
    outputs = model(perturbed_images)
    labels = torch.argmax(outputs, dim=-1)
    mask = labels == 8
    success = 100* len(labels[mask]) / len(labels)
    import pdb
    pdb.set_trace()


    net = Autoencoder()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    epsilon_large = 16
    for i in range(1000):
        model.zero_grad()
        perturbed_images = images_to_attack.detach().clone()
        # perturbed_images = images_to_attack.detach().clone().reshape(images_to_attack.shape[0], -1)
        perturbed_images = net(perturbed_images/255)
        perturbed_images = perturbed_images * 255
        # perturbed_images = perturbed_images.reshape(perturbed_images.shape[0], 1,28, 28)
        perturbed_images = torch.clamp(perturbed_images, min=images_to_attack-epsilon_large, max=images_to_attack+epsilon_large)
        perturbed_images = torch.clamp(perturbed_images, 0, 255)
        output = model(perturbed_images)
        loss = F.cross_entropy(output, target_labels)
        print(f"Iters: {i}: {loss}")
        loss.backward()
        optimizer.step()
    perturbed_images = images_to_attack.detach().clone()
    # perturbed_images = images_to_attack.detach().clone().reshape(images_to_attack.shape[0], -1)
    perturbed_images = net(perturbed_images/255)
    perturbed_images = perturbed_images * 255
    # perturbed_images = perturbed_images.reshape(perturbed_images.shape[0], 1,28, 28)
    perturbed_images = torch.clamp(perturbed_images, min=images_to_attack-epsilon_large, max=images_to_attack+epsilon_large)
    perturbed_images = torch.clamp(perturbed_images, 0, 255)
    outputs = model(perturbed_images)
    labels = torch.argmax(outputs, dim=-1)
    mask = labels == 8
    success = (labels[mask].sum() / len(labels)).item()
    import pdb
    pdb.set_trace()
    

    loss = F.cross_entropy(outputs, torch.tensor([8]))
    loss.backward()            
    optimizer.step()
    label = torch.argmax(outputs).item()
    



    mask_1 = labels == 1

    images_to_attack = images[mask_1].detach().clone()
    labels_to_attack = labels[mask_1].detach().clone()
    perturbed_images = images_to_attack.detach().clone()
    target_labels = torch.full_like(labels_to_attack, target_class)

    # perturbed_images = perturbed_images + chosen_image_8
    perturbed_images = torch.clamp(perturbed_images, 0, 255)

    m=torch.zeros(perturbed_images.shape)
    v=torch.zeros(perturbed_images.shape)
    epsilon_large = 128

    for i in range(iters): # PGD
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, target_labels)
        model.zero_grad()
        loss.backward()
        grads = perturbed_images.grad
        with torch.no_grad():
            t=i+1
            m=0.9*m+0.1*grads
            v=0.999*v+0.001*grads*grads
            mhat=m/(1.0 - 0.9**t)
            vhat=v/(1.0 - 0.999**t)
            grads=mhat / (torch.sqrt(vhat) + 1e-8)

            perturbed_images = perturbed_images - alpha * grads.sign()
            perturbed_images = torch.clamp(perturbed_images, min=images_to_attack-epsilon_large, max=images_to_attack+epsilon_large)
            perturbed_images = torch.clamp(perturbed_images, 0, 255)
            perturbed_images = perturbed_images.detach().clone()

    perturbed_images.requires_grad = False
    return_images = images.detach().clone()
    return_images[mask_1] = perturbed_images

    # mask_1 = labels == 1
    # images_to_attack = images[mask_1].detach().clone()
    # labels_to_attack = labels[mask_1].detach().clone()
    # perturbed_images = images_to_attack.detach().clone()
    # perturbations = []
    # labels = []
    # epsilon_large = 128
    # attack_perturbation = {}
    # for i in range(images_to_attack.shape[0]):
    #     print(f"Image:{i}")
    #     image = images_to_attack[i]

    #     perturbation = torch.zeros((1, images.shape[1], images.shape[2], images.shape[3]), requires_grad=True)
    #     optimizer = torch.optim.Adam([perturbation], lr=1)
    #     for _ in range(100):
    #         optimizer.zero_grad()
            
    #         # Apply perturbations within epsilon constraints
    #         perturbed_images = image + perturbation
    #         perturbed_images = torch.clamp(perturbed_images, 0, 255)
    #         perturbed_images = torch.clamp(perturbed_images - image, min=-epsilon_large, max=epsilon_large) + image
            
    #         outputs = model(perturbed_images)
    #         loss = F.cross_entropy(outputs, torch.tensor([8]))
    #         loss.backward()            
    #         optimizer.step()
    #         label = torch.argmax(outputs).item()

    #         # Project the perturbations to ensure they are within the epsilon ball after update
    #         perturbation.data = torch.clamp(perturbation, min=-epsilon_large, max=epsilon_large)
    #         if label == 8:
    #             attack_perturbation[i] = perturbation.detach().clone()
    #             break
    # attack_success_amt = len(attack_perturbation)
    # print(f"Epsilon:{epsilon_large} -> {attack_success_amt}")

    return_images_list = []
    for epsilon in epsilons:
        print(f"Improved target attack: Running on epsilon:{epsilon}")
        perturbed_images = images_to_attack.detach().clone()
        for i in range(perturbed_images.shape[0]):
            temp_image = perturbed_images[i].detach().clone()
            if i in attack_perturbation:
                perturbation = attack_perturbation[i]
                for _ in range(iters):
                    temp_image = temp_image + perturbation * 0.1
                    temp_image = torch.clamp(temp_image - perturbed_images[i].detach().clone(), min=-epsilon, max=epsilon) + perturbed_images[i].detach().clone()
                    temp_image = torch.clamp(temp_image, 0, 255)
                    
                    outputs = model(temp_image)
                    label = torch.argmax(outputs).item()

                    if label == 8:
                        perturbed_images[i] = temp_image.detach().clone()
                        break
        
        return_images = images.detach().clone()
        return_images[mask_1] = perturbed_images
        return_images_list.append(return_images)

    return return_images_list


    


    # mask_1 = labels == 1
    # mask_8 = labels == 8
    # images_label_8 = images[mask_8]
    # # chosen_image_8 = images_label_8[torch.randint(len(images_label_8), (1,))].clone()
    

    # images_to_attack = images[mask_1].detach().clone()
    # labels_to_attack = labels[mask_1].detach().clone()
    # perturbed_images = images_to_attack.detach().clone()
    # target_labels = torch.full_like(labels_to_attack, target_class)

    # # perturbed_images = perturbed_images + chosen_image_8
    # perturbed_images = torch.clamp(perturbed_images, 0, 255)

    # m=torch.zeros(perturbed_images.shape)
    # v=torch.zeros(perturbed_images.shape)

    # for i in range(iters): # PGD
    #     perturbed_images.requires_grad = True
    #     outputs = model(perturbed_images)
    #     loss = F.cross_entropy(outputs, target_labels)
    #     model.zero_grad()
    #     loss.backward()
    #     grads = perturbed_images.grad
    #     with torch.no_grad():
    #         t=i+1
    #         m=0.9*m+0.1*grads
    #         v=0.999*v+0.001*grads*grads
    #         mhat=m/(1.0 - 0.9**t)
    #         vhat=v/(1.0 - 0.999**t)
    #         grads=mhat / (torch.sqrt(vhat) + 1e-8)

    #         perturbed_images = perturbed_images - alpha * grads.sign()
    #         perturbed_images = torch.clamp(perturbed_images, min=images_to_attack-epsilon, max=images_to_attack+epsilon)
    #         perturbed_images = torch.clamp(perturbed_images, 0, 255)
    #         perturbed_images = perturbed_images.detach().clone()

    # perturbed_images.requires_grad = False
    # return_images = images.detach().clone()
    # return_images[mask_1] = perturbed_images

    return return_images

#evaluate performance of attacks
	#TODO

# models = [modelA, modelB]
models = [modelA]
model_names = ['Model B']
# model_names = ['Model A', 'Model B']
attacks = [targeted_attack_improved]
# attacks = [untargeted_attack, targeted_attack, targeted_attack_improved]
attack_names = ['Improved Targeted Attack']
# attack_names = ['Untargeted Attack', 'Targeted Attack', 'Improved Targeted Attack']
# epsilons = [i for i in range(1, 17)]
# epsilons = [4,8]
epsilons = [16]

# Placeholder for success rates
success_rates = np.zeros((len(models), len(attacks), len(epsilons)))

for i, model in enumerate(models):
    print(f"Model:{model_names[i]}")
    for j, attack in enumerate(attacks):
        print(f"Attack:{attack_names[j]}")
        if attack_names[j] != "Improved Targeted Attack":
            for k, epsilon in enumerate(epsilons):
                print(f"Epsilon: {epsilon}")
                if attack_names[j] != "Untargeted Attack":
                    adv_examples = attack(model=model, images=x_test_tensor, labels=y_test_tensor, target_class= 8, epsilon=epsilon)
                    success_rate = evaluate_attack_success_rate(model, adv_examples, y_test_tensor, target_class=8, attack_class=1)
                else:
                    adv_examples = attack(model=model, images=x_test_tensor, labels=y_test_tensor, epsilon=epsilon)
                    success_rate = evaluate_attack_success_rate(model, adv_examples, y_test_tensor)

                mask = y_test_tensor == 1
                original_images = x_test[mask][:3]
                adversarial_images = adv_examples[mask][:3]

                plot_adversarial_examples(original_images, adversarial_images, 3, epsilon, model_names[i], attack_names[j])
                test(model, adv_examples, y_test, 512)
                success_rates[i, j, k] = success_rate
                print(f"Attack Success Rate:{success_rate}")
        else:
            all_adv_examples = attack(model=model, images=x_test_tensor, labels=y_test_tensor, target_class= 8, epsilons=epsilons)
            for k, epsilon in enumerate(epsilons):
                adv_examples = all_adv_examples[k]
                success_rate = evaluate_attack_success_rate(model, adv_examples, y_test_tensor, target_class=8, attack_class=1)

                mask = y_test_tensor == 1
                original_images = x_test[mask][:3]
                adversarial_images = adv_examples[mask][:3]
                plot_adversarial_examples(original_images, adversarial_images, 3, epsilon, model_names[i], attack_names[j])
                test(model, adv_examples, y_test, 512)
                success_rates[i, j, k] = success_rate
                print(f"Attack Success Rate:{success_rate}")
    print("\n")

# Plotting
for i in range(len(models)):
    plt.figure(figsize=(10, 6))
    for j in range(len(attacks)):
        plt.plot(epsilons, success_rates[i, j, :], marker='o', label=attack_names[j])
        for x, y in zip(epsilons, success_rates[i, j, :]):
            plt.text(x, y, str(round(y, 4)), color="black", ha='center', va='bottom')
    plt.title(f'Success Rate vs Epsilon for {model_names[i]}')
    plt.xlabel('Epsilon')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_names[i]}.png")