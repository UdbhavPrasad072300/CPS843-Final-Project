import logging

import torch
import torch.nn as nn

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import ViT
from src.models.train import train
from src.visualization.visualize import plot_sequential
from src.models.test import test

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

    # Records

    test_acc = []

    # Get 4 ViT Models

    img_size = 32
    c_size = 3
    e_size = 256
    n_heads = 4
    classes = 10
    num_layers = 2
    hidden_size = 256
    dropout = 0.3

    criterion = nn.CrossEntropyLoss()

    # 1 Encoder

    print("-" * 20, "PATCH 16", "-" * 20)
    model = ViT(image_size=img_size,
                channel_size=c_size,
                patch_size=16,
                embed_size=e_size,
                num_heads=n_heads,
                classes=classes,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout
                ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_1 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_patch_16.pt')

    # 2 Encoder

    print("-" * 20, "PATCH 8", "-" * 20)
    model = ViT(image_size=img_size,
                channel_size=c_size,
                patch_size=8,
                embed_size=e_size,
                num_heads=n_heads,
                classes=classes,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout
                ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_2 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_patch_8.pt')

    # 3 Encoder

    print("-" * 20, "PATCH 4", "-" * 20)
    model = ViT(image_size=img_size,
                channel_size=c_size,
                patch_size=4,
                embed_size=e_size,
                num_heads=n_heads,
                classes=classes,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout
                ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_3 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_patch_4.pt')

    # 4 Encoder

    # print("-" * 20, "PATCH 2", "-" * 20)
    # model = ViT(image_size=img_size,
    #            channel_size=c_size,
    #            patch_size=2,
    #            embed_size=e_size,
    #            num_heads=n_heads,
    #            classes=classes,
    #            num_layers=num_layers,
    #            hidden_size=hidden_size,
    #            dropout=dropout
    #            ).to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    # loss_hist_4 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    # test_acc.append(test(model, test_loader, DEVICE))
    # torch.save(model.state_dict(), './trained_models/vit_patch_2.pt')

    # Plot Train Stats

    plot_sequential(loss_hist_1["train accuracy"], "Patch 16 - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_1["train loss"], "Patch 16 - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_1["val accuracy"], "Patch 16 - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_2["train accuracy"], "Patch 8 - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_2["train loss"], "Patch 8 - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_2["val accuracy"], "Patch 8 - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_3["train accuracy"], "Patch 4 - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_3["train loss"], "Patch 4 - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_3["val accuracy"], "Patch 4 - ViT", "Epoch", "Validation Accuracy")

    #plot_sequential(loss_hist_4["train accuracy"], "Patch 2 - ViT", "Epoch", "Train Accuracy")
    #plot_sequential(loss_hist_4["train loss"], "Patch 2 - ViT", "Epoch", "Train Loss")
    #plot_sequential(loss_hist_4["val accuracy"], "Patch 2 - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(test_acc, "Test Accuracies - Patches 16-8-4 - ViT", "Encoder Num", "Test Accuracy")

    # [52.22, 57.96, 58.06]
    print(test_acc)

    print("Program has Ended")
