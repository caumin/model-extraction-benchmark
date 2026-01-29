
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from mebench.models.substitute_factory import LeNetMNIST, ResNet

# --- Configuration ---
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "runs/victims"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / total, 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / total, 100. * correct / total

def train_mnist_lenet():
    print("\n=== Training MNIST LeNet ===")
    
    # 1. Data (No Normalization, [0, 1] range)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    
    # 2. Model
    model = LeNetMNIST(num_classes=10, input_channels=1).to(DEVICE)
    
    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop (20 Epochs)
    best_acc = 0.0
    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save strictly the state_dict
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mnist_lenet_mnist_seed0.pt"))
            
    print(f"Best MNIST Accuracy: {best_acc:.2f}%")

def train_cifar10_resnet18():
    print("\n=== Training CIFAR10 ResNet18 ===")
    
    # 1. Data (No Mean/Std Normalization, but Augmentation applied)
    # Note: Augmentations like RandomCrop/Flip don't change the value range/distribution scale significantly like Normalize does.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    
    # 2. Model
    model = ResNet(num_classes=10, width_mult=1, input_channels=3).to(DEVICE)
    
    # 3. Optimization (Standard SGD Recipe)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 4. Training Loop (200 Epochs)
    best_acc = 0.0
    # Use tqdm for long training
    pbar = tqdm(range(200), desc="CIFAR10 Training")
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        scheduler.step()
        
        pbar.set_postfix({"Train Acc": f"{train_acc:.2f}%", "Test Acc": f"{test_acc:.2f}%"})
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cifar10_resnet18_seed0.pt"))
            
    print(f"Best CIFAR10 Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Train MNIST
    train_mnist_lenet()
    
    # Train CIFAR10
    train_cifar10_resnet18()
