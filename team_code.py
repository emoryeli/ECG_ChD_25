#!/usr/bin/env python

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.signal import resample_poly  # for resampling ECG signals to 400Hz
from helper_code import *

# Select device (GPU or MPS or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()  # helpful after large model or dataset loads
elif device.type == 'cpu':
    torch.set_num_threads(os.cpu_count())  # or 16

THRESHOLD_PROBABILITY = 0.5 # above this threshold, the model predicts Chagas disease, add threshold optimization later
source_string = '# Source:' # used to remove CODE 15% data from the traiing set
CODE15 = 'CODE-15%' # used to remove CODE 15% data from the traiing set
max_len = 4096 # (10.2s at 400Hz) must be power of 2 because of the 1d ResNet model from Screening for Chagas paper, 2020

class ECGDataset(Dataset):
    def __init__(self, records, labels):
        self.records = records
        self.labels = labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal = extract_ECG(self.records[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the training data...')

    # Find the records and remove CODE 15% data from the training set
    all_records = find_records(data_folder)
    records = []
    for record in all_records:
        header = load_header(os.path.join(data_folder, record))
        source, has_source = get_variable(header, source_string)
        Chagas_label, has_label = get_variable(header, label_string)
        #if Chagas_label == 'True' or source != CODE15: # remove chagas positive cases in CODE-15% dataset
        if source != CODE15: # remove all CODE-15% data (this may pose some risk if the competition data set has also weakly labeled data)
            records.append(record)

    if len(records) == 0:
        raise FileNotFoundError('No useful data were provided after removing CODE 15 data.')

    if verbose:
        print('Extracting labels...')

    labels = [load_label(os.path.join(data_folder, rec)) for rec in records]
    data_paths = [os.path.join(data_folder, r) for r in records]
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_val_loss_overall = float('inf')
    best_model_state_overall = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels), 1):
        if verbose:
            print(f"\nStarting Fold {fold}...")

        train_records = [data_paths[i] for i in train_idx]
        val_records = [data_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_dataset = ECGDataset(train_records, train_labels)
        val_dataset = ECGDataset(val_records, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        model = ResNet1D_Chagas().to(device).to(torch.float32) # use float32 for training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 20 epochs
        #loss_fn = nn.CrossEntropyLoss()
        weight = torch.tensor([1.0, 19.0])  # assuming 5% positive -> 1:19 imbalance
        loss_fn = nn.CrossEntropyLoss(weight=weight.to(device)) # , label_smoothing=0.1 for weakly labeled data

        best_val_loss = float('inf')

        for epoch in range(20):
            # if verbose:
            #     print(f"\nStarting Fold {fold}, Epoch {epoch + 1}:")

            model.train()
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X.to(device))
                loss = loss_fn(outputs, y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping (for long ECGs)
                optimizer.step()
                total_loss += loss.item() * X.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0
            val_outputs = [] # predicted probabilities, float32
            val_targets = [] # labels, 0 or 1
            
            with torch.no_grad():
                for X, y in val_loader:
                    outputs = model(X.to(device))
                    loss = loss_fn(outputs, y.to(device))
                    val_loss += loss.item() * X.size(0)

                    #probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # probs for class 1: Chagas; .cpu().numpy() is slower
                    probs = torch.softmax(outputs, dim=1)[:, 1].to(dtype=torch.float32)
                    if probs.device.type != 'cpu':
                        probs = probs.cpu()
                    probs = probs.numpy()

                    val_outputs.extend(probs.tolist())
                    val_targets.extend(y.tolist())
                    #val_targets.extend(y.cpu().numpy().tolist()) # slower

            avg_val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(avg_val_loss) # scheduler.step() if using CosineAnnealingLR
            f1 = compute_f_measure(val_targets, (np.array(val_outputs) > THRESHOLD_PROBABILITY).astype(int))
            challenge_score = compute_challenge_score(np.array(val_targets), np.array(val_outputs))
            auroc, auprc = compute_auc(np.array(val_targets), np.array(val_outputs))

            if verbose:
                print(f"Fold {fold}, Epoch {epoch + 1}: "
                    f"Train Loss = {avg_train_loss:.2e}, "
                    f"Validation Loss = {avg_val_loss:.2e}, "
                    f"F1 = {f1:.4f}, "
                    f"AUROC = {auroc:.4f}, "
                    f"AUPRC = {auprc:.4f}, "
                    f"Challenge Score = {challenge_score:.4f}, "
                    f"LR = {optimizer.param_groups[0]['lr']:.2e}")
                
                # Check how many true positives are in top 5%
                val_targets = np.array(val_targets)
                val_outputs = np.array(val_outputs)

                # sort by predicted probability and get top 5% indices and labels
                #top5_indices = np.argsort(val_outputs)[::-1][:int(0.05 * len(val_outputs))]
                #top5_labels = val_targets[top5_indices]

                #print(f"Chagas instances among Top 5% probabilities: {np.sum(top5_labels)}. Total ground truth positives: {np.sum(val_targets)}.")
                #print(len(val_outputs))
                #print(len(val_targets))
                #print(f"top 5% indices: {top5_indices}")
                #print(f"top 5% labels: {top5_labels}")
                #all_indices = np.argsort(val_outputs)[::-1][:int(len(val_outputs))]
                #print("orded labels:", val_targets[all_indices])
                #print("predicted probabilities:", np.sort(val_outputs)[::-1])
                #print('Fold', fold, 'Epoch', epoch + 1, 'Ended')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        if best_val_loss < best_val_loss_overall:
            best_val_loss_overall = best_val_loss
            best_model_state_overall = best_model_state

    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': best_model_state_overall}, os.path.join(model_folder, 'model.pt'))

    if verbose:
        print(f"\nBest model saved with validation loss: {best_val_loss_overall:.4f}")

def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_folder, 'model.pt'))

def load_model(model_folder, verbose):
    model = ResNet1D_Chagas()
    checkpoint = torch.load(os.path.join(model_folder, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model

def run_model(record, model, verbose):
    x = extract_ECG(record).unsqueeze(0) # (1, 12, max_len)
    x = x.to(next(model.parameters()).device) # moves x to the same device that the model's parameters are on
    output = model(x)
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    binary_output = int(probs[1] > THRESHOLD_PROBABILITY)  # 1 for Chagas disease, 0 for no Chagas diseas
    return binary_output, probs[1]

def extract_ECG(record):
    # Load signal as (samples, leads), and header as text string
    signal, _ = load_signals(record)
    header = load_header(record)

    # Get sampling frequency
    fs = get_sampling_frequency(header)
    if fs is None:
        fs = 500  # Default, if not present in header

    signal = np.nan_to_num(signal).T  # Now shape is (leads, samples)

    # Resample to 400 Hz using polyphase filtering
    target_fs = 400
    if fs != target_fs:
        # Compute upsample and downsample factors
        from math import gcd
        g = gcd(int(target_fs), int(fs))
        up = target_fs // g
        down = fs // g

        # Resample each lead separately using resample_poly
        signal = np.array([resample_poly(lead, up, down) for lead in signal])

    # Truncate or pad to fixed length (e.g. max_len = 2934 samples for 7.2s at 400 Hz)
    # max_len = 2934 or 4096 or 5000, some SaMi and CODE-15% data have 2934 samples
    if signal.shape[1] < max_len:
        pad_width = max_len - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)))
    else:
        signal = signal[:, :max_len]

    return torch.tensor(signal, dtype=torch.float32)

# ---------
# Adapted code from github repo: https://github.com/antonior92/ecg-chagas/

def _padding(downsample, kernel_size):
    "Compute required padding"
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding

def _downsample(n_samples_in, n_samples_out):
    "Compute downsample rate"
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample

class ResBlock1d(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()

        # Forward path
        padding1 = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        padding2 = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding2, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []

        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]

        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]

        # Build skip connection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        "Residual unit"
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y

        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y

class ResNet1D_Chagas(nn.Module): # 9,626,386 trainable parameters
    # Residual network for 1d ECG signals.

    def __init__(self, input_dim = (12, max_len), blocks_dim = list(zip([64, 128, 256, 512],[max_len, max_len//2, max_len//4, max_len//8])), n_classes=2, kernel_size=17, dropout_rate=0.5):
        super(ResNet1D_Chagas, self).__init__()

        # First layer
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]  # 12, 64
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]  # 4096, 4096
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{nr}'.format(nr=i), resblk1d)  # make the resblocks actual modules (self.resblock_0 etc)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        print(last_layer_dim)

        # number of residual blocks
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.lin(x)
        return x
