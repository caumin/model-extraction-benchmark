import os
import sys
from pathlib import Path

# Add project root to path to allow importing mebench
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from mebench.models.substitute_factory import LeNetMNIST

def train():
    # Set seeds for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = project_root / "runs" / "victims"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mnist_lenet_seed0.pt"

    # Data loaders
    # LeNetMNIST expects 28x28 inputs (standard MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download to project root's data directory
    data_root = project_root / "data"
    
    print("Loading datasets...")
    train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize Model
    print("Initializing LeNetMNIST...")
    model = LeNetMNIST(num_classes=10, input_channels=1).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    print(f"Starting training for {epochs} epochs...")
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}%")

    # Save model
    print(f"Saving model to {output_path}")
    torch.save(model.state_dict(), output_path)
    print("Done.")

if __name__ == "__main__":
    train()
