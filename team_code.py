#!/usr/bin/env python

import joblib
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from helper_code import *

# Select device (CPU or GPU)
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
THRESHOLD_PROBABILITY = 0.5

class ECGDataset(Dataset):
    def __init__(self, records, labels):
        self.records = records
        self.labels = labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal = extract_features(self.records[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the training data...')

    records = find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Extracting labels...')

    labels = [load_label(os.path.join(data_folder, rec)) for rec in records]
    data_paths = [os.path.join(data_folder, r) for r in records]
    labels = np.array(labels)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_val_loss_overall = float('inf')
    best_model_state_overall = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_paths)):
        if verbose:
            print(f"\nStarting Fold {fold + 1}...")

        train_records = [data_paths[i] for i in train_idx]
        val_records = [data_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_dataset = ECGDataset(train_records, train_labels)
        val_dataset = ECGDataset(val_records, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = ECG1DResNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float('inf')

        for epoch in range(30):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X.to(device))
                loss = loss_fn(outputs, y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0
            val_outputs = [] #predicted probabilities, float64
            val_targets = [] #labels, 0 or 1
            
            with torch.no_grad():
                for X, y in val_loader:
                    outputs = model(X.to(device))
                    loss = loss_fn(outputs, y.to(device))
                    val_loss += loss.item() * X.size(0)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # probs for class 1: Chagas
                    val_outputs.extend(probs.tolist())
                    val_targets.extend(y.cpu().numpy().tolist())
            avg_val_loss = val_loss / len(val_loader.dataset)
            f1 = compute_f_measure(val_targets, (np.array(val_outputs) >= THRESHOLD_PROBABILITY).astype(int))
            challenge_score = compute_challenge_score(np.array(val_targets), np.array(val_outputs))
            scheduler.step(avg_val_loss)

            if verbose:
                print(f"Fold {fold + 1}, Epoch {epoch + 1}: "
                    f"Train Loss = {avg_train_loss:.4f}, "
                    f"Val Loss = {avg_val_loss:.4f}, "
                    f"F1 = {f1:.4f}, "
                    f"Challenge Score = {challenge_score:.4f}, "
                    f"LR = {optimizer.param_groups[0]['lr']:.2e}")
                
                # Check how many true positives are in top 5%
                #val_targets = np.array(val_targets)
                #val_outputs = np.array(val_outputs)

                # sort by predicted probability
                #top5_indices = np.argsort(val_outputs)[::-1][:int(0.05 * len(val_outputs))]
                #top5_labels = val_targets[top5_indices]

                #print(f"Top 5% Chagas cases: {np.sum(top5_labels)} out of {np.sum(val_targets)} positives")
                #print(len(val_outputs))
                #print(len(val_targets))
                #print(f"top 5% indices: {top5_indices}")
                #print(f"top 5% labels: {top5_labels}")
                all_indices = np.argsort(val_outputs)[::-1][:int(len(val_outputs))]
                #print("orded labels:", val_targets[all_indices])
                #print("predicted probabilities:", np.sort(val_outputs)[::-1])

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
    model = ECG1DResNet()
    checkpoint = torch.load(os.path.join(model_folder, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_model(record, model, verbose):
    x = extract_features(record).unsqueeze(0)
    output = model(x)
    probs = torch.softmax(output, dim=1).detach().numpy()[0]
    binary_output = int(probs[1] > THRESHOLD_PROBABILITY)  # 1 for Chagas disease, 0 for no Chagas diseas
    return binary_output, probs[1]

def extract_features(record):
    signal, _ = load_signals(record)
    signal = np.nan_to_num(signal).T
    max_len = 5000 # ptb-xl data has 5000 samples
    if signal.shape[1] < max_len:
        pad_width = max_len - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)))
    else:
        signal = signal[:, :max_len]
    return torch.tensor(signal, dtype=torch.float32)

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=17, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1, stride=stride) if downsample or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class ECG1DResNet(nn.Module):
    def __init__(self, input_channels=12, num_classes=2):
        super().__init__()
        self.block1 = ResidualBlock(input_channels, 32, downsample=True)
        self.block2 = ResidualBlock(32, 64, downsample=True)
        self.block3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
