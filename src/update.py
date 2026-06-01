#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attacks import (
    BACKDOOR, LABEL_FLIP, apply_data_attack, sample_backdoor_indices,
    stamp_backdoor_trigger,
)
from utils import get_device, get_logger, get_optimizer


LOGGER = get_logger(__name__)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, return_index=False):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.return_index = return_index

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        label = torch.as_tensor(label)
        if self.return_index:
            return image, label, torch.as_tensor(item)
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id=None,
                 is_malicious=False):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.idxs = list(idxs)
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.local_dataset = DatasetSplit(dataset, self.idxs)
        self.trainloader = self.build_trainloader()
        self.testloader = None
        self.device = get_device(args)
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss()

    def build_trainloader(self):
        """Build a train loader over all local data for one client."""
        train_dataset = DatasetSplit(self.dataset, self.idxs,
                                     return_index=True)
        return DataLoader(train_dataset, batch_size=self.args.local_bs,
                          shuffle=True)

    def build_testloader(self):
        """Build the per-client eval loader only when inference needs it."""
        eval_batch_size = max(1, min(len(self.local_dataset),
                                     self.args.local_bs))
        return DataLoader(self.local_dataset, batch_size=eval_batch_size,
                          shuffle=False)

    def update_weights(self, model, global_round):
        # Set mode to train model
        # Federated training passes copied models, so move each copy explicitly.
        model.to(self.device)
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = get_optimizer(self.args, model)
        backdoor_indices = sample_backdoor_indices(
            self.args, len(self.local_dataset), self.is_malicious)

        for iter in range(self.args.local_ep):
            running_loss, num_seen = 0.0, 0
            for batch_idx, batch in enumerate(self.trainloader):
                if len(batch) == 3:
                    images, labels, local_indices = batch
                else:
                    images, labels = batch
                    local_indices = None
                images, labels = images.to(self.device), labels.to(self.device)
                images, labels = apply_data_attack(
                    self.args, images, labels, self.is_malicious,
                    local_indices=local_indices,
                    backdoor_indices=backdoor_indices)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                running_loss += loss.item() * labels.size(0)
                num_seen += labels.size(0)
            epoch_loss.append(running_loss / num_seen)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """Returns accuracy and loss on one client's full local partition."""

        model.eval()
        # Keep inference on the same device used for local training.
        model.to(self.device)
        if self.testloader is None:
            self.testloader = self.build_testloader()
        loss, total, correct = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item() * labels.size(0)

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += labels.size(0)

        accuracy = correct/total
        return accuracy, loss/total


def test_inference(args, model, test_dataset):
    """Returns accuracy and loss on the global test dataset.

    This evaluates the model against the official dataset test split, so it is
    used for centralized baseline metrics and final/global federated metrics.
    """

    model.eval()
    loss, total, correct = 0.0, 0, 0

    # Evaluate on the device where the caller placed the model.
    device = next(model.parameters()).device
    criterion = nn.NLLLoss()
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * labels.size(0)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += labels.size(0)

    accuracy = correct/total
    return accuracy, loss/total


def test_attack_success_rate(args, model, test_dataset):
    """Evaluate targeted attack success rate on the global test dataset."""
    if args.attack not in (LABEL_FLIP, BACKDOOR):
        return None

    model.eval()
    device = next(model.parameters()).device
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    target_label = int(args.attack_target_label)
    total, success = 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            if args.attack == LABEL_FLIP:
                mask = labels == int(args.label_flip_source)
                if not torch.any(mask).item():
                    continue
                attack_images = images[mask]
            else:
                mask = labels != target_label
                if not torch.any(mask).item():
                    continue
                attack_images = stamp_backdoor_trigger(args, images[mask])

            outputs = model(attack_images)
            _, pred_labels = torch.max(outputs, 1)
            success += torch.sum(pred_labels == target_label).item()
            total += pred_labels.numel()

    if total == 0:
        return 0.0
    return success / total
