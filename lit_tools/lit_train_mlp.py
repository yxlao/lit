import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# Define the MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for first layer
        x = F.relu(self.fc2(x))  # Activation function for second layer
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for output layer
        return x


def load_and_split_dataset(file_path, split_ratio=0.8):
    data = np.load(file_path)
    network_inputs = data["network_inputs"]
    network_outputs = data["network_outputs"]

    split_index = int(len(network_inputs) * split_ratio)
    train_inputs = network_inputs[:split_index]
    train_outputs = network_outputs[:split_index]
    val_inputs = network_inputs[split_index:]
    val_outputs = network_outputs[split_index:]

    return train_inputs, train_outputs, val_inputs, val_outputs


def prepare_dataloader(inputs, outputs, batch_size=512):
    tensor_x = torch.Tensor(inputs)
    tensor_y = torch.Tensor(outputs)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        step = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            step += 1
            if step % 1000 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.round()  # Threshold at 0.5
            total += targets.size(0)
            correct += (predicted == targets.unsqueeze(1)).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    return accuracy


# Main function
def main():
    # Load and prepare data
    train_inputs, train_outputs, val_inputs, val_outputs = load_and_split_dataset(
        Path.home() / "research/lit/data/nuscenes/09_raykeep/raykeep_data.npz"
    )
    train_loader = prepare_dataloader(train_inputs, train_outputs, batch_size=512)
    val_loader = prepare_dataloader(val_inputs, val_outputs, batch_size=512)

    # Device configuration
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, Loss, and Optimizer
    input_size = train_inputs.shape[1]
    model = SimpleMLP(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and Evaluate
    train(model, train_loader, criterion, optimizer, num_epochs=10)
    evaluate(model, val_loader)


if __name__ == "__main__":
    main()
