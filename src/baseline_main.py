#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import (
    get_dataset, get_device, get_logger, get_optimizer, get_run_name, log_args,
    log_git_commit, set_seed, format_run_time,
)
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet18Cifar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_ROOT / 'save'
LOGGER = get_logger(__name__)


if __name__ == '__main__':
    start_time = time.time()
    log_git_commit('begin', LOGGER)

    args = args_parser()
    # Prefer CUDA when explicitly requested; otherwise use MPS if available.
    device = get_device(args)
    # Cache the resolved device so LocalUpdate/test helpers do not re-detect it.
    args.device = str(device)
    args.seed = set_seed(args.seed)
    LOGGER.info('Using device: %s', device)
    log_args(args)
    run_name = get_run_name(
        args,
        'baseline',
        ['dataset', 'model', 'epochs', 'optimizer', 'lr', 'batch_size',
         'scheduler', 'test_interval'],
    )

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
    # LOGGER.info('%s', global_model)

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
    test_epochs, test_accuracy, test_losses = [], [], []

    for epoch in range(args.epochs):
        global_model.train()
        running_loss, num_seen = 0.0, 0

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
            running_loss += loss.item() * labels.size(0)
            num_seen += labels.size(0)

        loss_avg = running_loss / num_seen
        LOGGER.info('\nTrain loss: %s', loss_avg)
        epoch_loss.append(loss_avg)
        if scheduler is not None:
            scheduler.step()

        current_epoch = epoch + 1
        should_test = (
            args.test_interval > 0 and
            (current_epoch % args.test_interval == 0 or
             current_epoch == args.epochs)
        )
        if should_test:
            test_acc, test_loss = test_inference(
                args, global_model, test_dataset)
            test_epochs.append(current_epoch)
            test_accuracy.append(test_acc)
            test_losses.append(test_loss)
            LOGGER.info(
                'Test after epoch %s/%s: Loss: %.4f | Accuracy: %.2f%%',
                current_epoch, args.epochs, test_loss, 100*test_acc)

    if not test_epochs or test_epochs[-1] != args.epochs:
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_epochs.append(args.epochs)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)
    else:
        test_acc = test_accuracy[-1]
        test_loss = test_losses[-1]

    # Plot loss
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(epoch_loss)+1), epoch_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.title('Training Loss')
    plt.tight_layout()
    loss_plot_path = SAVE_DIR / f'{run_name}_loss.png'
    plt.savefig(loss_plot_path)
    plt.close()
    LOGGER.info('Saved loss figure: %s', loss_plot_path)

    # Plot accuracy
    test_accuracy_percent = [100*acc for acc in test_accuracy]

    plt.figure()
    plt.plot(test_epochs, test_accuracy_percent, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy (%)')
    plt.title('Test Accuracy')
    plt.tight_layout()
    acc_plot_path = SAVE_DIR / f'{run_name}_acc.png'
    plt.savefig(acc_plot_path)
    plt.close()
    LOGGER.info('Saved accuracy figure: %s', acc_plot_path)

    LOGGER.info('Train loss array by epoch: %s', epoch_loss)
    LOGGER.info('Test epochs array: %s', test_epochs)
    LOGGER.info('Test accuracy array by test epoch: %s', test_accuracy)
    LOGGER.info('Test accuracy percent array by test epoch: %s',
                test_accuracy_percent)
    LOGGER.info('Test loss array by test epoch: %s', test_losses)

    LOGGER.info('Test on %s samples', len(test_dataset))
    LOGGER.info("Test Accuracy: {:.2f}%".format(100*test_acc))
    LOGGER.info('\n Total Run Time: %s', format_run_time(time.time()-start_time))
    log_git_commit('end', LOGGER)
