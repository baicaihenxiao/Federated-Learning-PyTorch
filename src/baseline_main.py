#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset, get_device, get_logger, get_optimizer, log_args
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet18Cifar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_ROOT / 'save'
LOGGER = get_logger(__name__)


if __name__ == '__main__':
    args = args_parser()
    # Prefer CUDA when explicitly requested; otherwise use MPS if available.
    device = get_device(args)
    # Cache the resolved device so LocalUpdate/test helpers do not re-detect it.
    args.device = str(device)
    LOGGER.info('Using device: %s', device)
    log_args(args)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'resnet18':
        if args.dataset != 'cifar':
            exit('Error: resnet18 is configured for cifar')
        global_model = ResNet18Cifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    LOGGER.info('%s', global_model)

    # Training
    # Set optimizer and criterion
    optimizer = get_optimizer(args, global_model)
    # Decay the learning rate over centralized training for better CIFAR tuning.
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    else:
        scheduler = None

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                LOGGER.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images),
                        len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        LOGGER.info('\nTrain loss: %s', loss_avg)
        epoch_loss.append(loss_avg)
        if scheduler is not None:
            scheduler.step()

    # Plot loss
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plot_path = SAVE_DIR / 'nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                    args.epochs)
    plt.savefig(plot_path)

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    LOGGER.info('Test on %s samples', len(test_dataset))
    LOGGER.info("Test Accuracy: {:.2f}%".format(100*test_acc))
