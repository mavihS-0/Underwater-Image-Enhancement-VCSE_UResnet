import torch
import argparse
from torchvision import transforms
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataloader import TrainDataSet

# Learnable Ensemble Model
class LearnableEnsembleModel(nn.Module):
    def __init__(self, models):
        super(LearnableEnsembleModel, self).__init__()
        self.models = models
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))  # Learnable weights

    def forward(self, x):
        outputs = [weight * model(x)[0] for model, weight in zip(self.models, self.weights)]
        ensemble_output = sum(outputs)
        return ensemble_output

# Training function
def train_ensemble(config):
    device = torch.device("cuda:" + str(config.cuda_id) if torch.cuda.is_available() else "cpu")

    # Load individual models
    model1 = torch.load(config.snapshot_pth1).to(device)
    model2 = torch.load(config.snapshot_pth2).to(device)
    model3 = torch.load(config.snapshot_pth3).to(device)

    # Freeze individual model parameters
    for model in [model1, model2, model3]:
        for param in model.parameters():
            param.requires_grad = False

    # Create learnable ensemble model
    ensemble_model = LearnableEnsembleModel([model1, model2, model3]).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss() if config.loss_type == 'MSE' else nn.L1Loss()
    optimizer = optim.Adam(ensemble_model.parameters(), lr=config.lr)

    # Transformations
    transform_list = [transforms.Resize((config.resize, config.resize)), transforms.ToTensor()]
    tsfms = transforms.Compose(transform_list)

    # Load dataset
    train_dataset = TrainDataSet(config.input_images_path, config.label_images_path, tsfms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Training loop
    loss_history = []
    for epoch in range(config.num_epochs):
        epoch_loss = []

        for input_img, label_img in train_dataloader:
            input_img, label_img = input_img.to(device), label_img.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = ensemble_model(input_img)
            loss = criterion(output, label_img)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        # Log epoch loss
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        loss_history.append(avg_loss)

        if epoch % config.print_freq == 0:
            print(f"Epoch {epoch}/{config.num_epochs}, Loss: {avg_loss}")

        # Save model checkpoints
        if not os.path.exists(config.snapshots_folder):
            os.makedirs(config.snapshots_folder)

        if epoch % config.snapshot_freq == 0:
            torch.save(ensemble_model, os.path.join(config.snapshots_folder, f'ensemble_model_epoch_{epoch}.ckpt'))

    return loss_history

# Plot loss curve
def plot_loss_curve(loss_hist):
    plt.plot(loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--snapshot_pth1', type=str, required=True, help='Path to snapshot of model1')
    parser.add_argument('--snapshot_pth2', type=str, required=True, help='Path to snapshot of model2')
    parser.add_argument('--snapshot_pth3', type=str, required=True, help='Path to snapshot of model3')
    parser.add_argument('--input_images_path', type=str, default="./data/input/",help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--label_images_path', type=str, default="./data/label/",help='path of label images(clear images) default:./data/label/')
    parser.add_argument('--snapshots_folder', type=str, default='./snapshots/', help='Folder to save snapshots')
    parser.add_argument('--resize', type=int, default=256, help='Resize dimension for images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='MSE', help='Loss type: MSE or L1')
    parser.add_argument('--print_freq', type=int, default=1, help='Frequency of printing logs')
    parser.add_argument('--snapshot_freq', type=int, default=10, help='Frequency of saving snapshots')

    config = parser.parse_args()
    loss_hist = train_ensemble(config)
    plot_loss_curve(loss_hist)

