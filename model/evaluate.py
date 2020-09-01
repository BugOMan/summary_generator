#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import pathlib
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from dataset import collate_fn
import config


def evaluate(model, val_data, epoch):
    """Evaluate the loss for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_data (dataset.PairDataset): The evaluation data set.
        epoch (int): The epoch number.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')

    val_loss = []
    with torch.no_grad():
        DEVICE = config.DEVICE
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    pin_memory=True, drop_last=True,
                                    collate_fn=collate_fn)
        for batch, data in enumerate(tqdm(val_dataloader)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            # Calculate loss.  Call model forward propagation
            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=len(val_dataloader), teacher_forcing=True)
            val_loss.append(loss.item())
    return np.mean(val_loss)
