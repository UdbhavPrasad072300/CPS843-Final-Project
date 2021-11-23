import logging

import torch
import torch.nn as nn

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import ViT
from src.models.train import train
from src.visualization.visualize import plot_sequential

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used: {}".format(DEVICE))

if __name__ == "__main__":
    # Get Data Loaders

    train_loader, val_loader, test_loader = get_dataloader("./data/CIFAR10/", config.BATCH_SIZE)

    print("Train Dataset Length: {}".format(len(train_loader)))
    print("Validation Dataset Length: {}".format(len(val_loader)))
    print("Test Dataset Length: {}".format(len(test_loader)))

    # Model Hyper-parameters

    image_size = 32
    channel_size = 3
    patch_size = 8
    embed_size = 512
    num_heads = 8
    classes = 100
    num_layers = 2
    hidden_size = 256
    dropout = 0.2

    # Instantiate Model

    model = ViT(image_size=image_size,
                channel_size=channel_size,
                patch_size=patch_size,
                embed_size=embed_size,
                num_heads=num_heads,
                classes=classes,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout).to(DEVICE)
    print(model)

    # Training

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    loss_hist = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)

    # Plot Train Stats

    plot_sequential(loss_hist["train accuracy"], "Train Accuracy", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist["train loss"], "Train Loss", "Epoch", "Train Loss")
    plot_sequential(loss_hist["val accuracy"], "Validation Accuracy", "Epoch", "Validation Accuracy")

    # Save

    torch.save(model.state_dict(), './trained_models/ViT_1.pt')

    print("Program has Ended")
