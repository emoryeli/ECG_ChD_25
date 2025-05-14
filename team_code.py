#!/usr/bin/env python

import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from helper_code import *

# Select device (GPU or MPS or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()  # helpful after large model or dataset loads

THRESHOLD_PROBABILITY = 0.5 # above this threshold, the model predicts Chagas disease
source_string = '# Source:' # used to remove CODE 15% data from the traiing set
CODE15 = 'CODE-15%' # used to remove CODE 15% data from the traiing set

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

    # Find the records and remove CODE 15% data from the training set
    all_records = find_records(data_folder)
    records = []
    for record in all_records:
        header = load_header(os.path.join(data_folder, record))
        source, has_source = get_variable(header, source_string)
        Chagas_label, has_label = get_variable(header, label_string)
        #if Chagas_label == 'True' or source != CODE15: # remove chagas negative data in CODE-15% dataset
        if source != CODE15: # remove all CODE-15% data
            records.append(record)

    if len(records) == 0:
        raise FileNotFoundError('No useful data were provided after removing CODE 15 data.')

    if verbose:
        print('Extracting labels...')

    labels = [load_label(os.path.join(data_folder, rec)) for rec in records]
    data_paths = [os.path.join(data_folder, r) for r in records]
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
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
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = ConvNeXtV2_1D_ECG().to(device).to(torch.float32) # use float32 for training
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        weight = torch.tensor([1.0, 19.0])  # 5% positive -> 1:19 imbalance
        loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))

        best_val_loss = float('inf')

        for epoch in range(20):
            if verbose:
                print(f"\nStarting Fold {fold}, Epoch {epoch + 1}: ")
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
            val_outputs = [] #predicted probabilities, float32
            val_targets = [] #labels, 0 or 1
            
            with torch.no_grad():
                for X, y in val_loader:
                    outputs = model(X.to(device))
                    loss = loss_fn(outputs, y.to(device))
                    val_loss += loss.item() * X.size(0)

                    probs = torch.softmax(outputs, dim=1)[:, 1].to(dtype=torch.float32)
                    if probs.device.type != 'cpu':
                        probs = probs.cpu()
                    probs = probs.numpy()

                    val_outputs.extend(probs.tolist())
                    val_targets.extend(y.tolist())
                    #val_targets.extend(y.cpu().numpy().tolist())

            avg_val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(avg_val_loss) # scheduler.step() if using CosineAnnealingLR
            f1 = compute_f_measure(val_targets, (np.array(val_outputs) > THRESHOLD_PROBABILITY).astype(int))
            challenge_score = compute_challenge_score(np.array(val_targets), np.array(val_outputs))
            auroc, auprc = compute_auc(np.array(val_targets), np.array(val_outputs))

            if verbose:
                print(f"Fold {fold}, Epoch {epoch + 1}: "
                    f"Train Loss = {avg_train_loss:.3e}, "
                    f"Validation Loss = {avg_val_loss:.3e}, "
                    f"F1 = {f1:.4f}, "
                    f"AUROC = {auroc:.4f}, "
                    f"AUPRC = {auprc:.4f}, "
                    f"Challenge Score = {challenge_score:.4f}, "
                    f"LR = {optimizer.param_groups[0]['lr']:.2e}")
                
                # Check how many true positives are in top 5%
                val_targets = np.array(val_targets)
                val_outputs = np.array(val_outputs)

                # sort by predicted probability and get top 5% indices and labels
                top5_indices = np.argsort(val_outputs)[::-1][:int(0.05 * len(val_outputs))]
                top5_labels = val_targets[top5_indices]

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
    model = model = ConvNeXtV2_1D_ECG()
    checkpoint = torch.load(os.path.join(model_folder, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model

def run_model(record, model, verbose):
    x = extract_features(record).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    output = model(x)
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    binary_output = int(probs[1] > THRESHOLD_PROBABILITY)  # 1 for Chagas disease, 0 for no Chagas diseas
    return binary_output, probs[1]

def extract_features(record):
    signal, _ = load_signals(record)
    signal = np.nan_to_num(signal).T
    max_len = 5000 # ptb-xl data has 5000 samples per ECG lead
    if signal.shape[1] < max_len:
        pad_width = max_len - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)))
    else:
        signal = signal[:, :max_len]
    return torch.tensor(signal, dtype=torch.float32)

# DropPath helper
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

# Global Response Normalization (GRN)
class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.eps   = eps
    def forward(self, x):                    # x shape (B, L, C)  after transpose
        g = torch.norm(x, p=2, dim=1, keepdim=True)  # global L2 norm across sequence per channel
        x = x / (g + self.eps)
        return self.gamma * x + self.beta
    
# 1D ConvNeXtV2 basic block
class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, dim, drop_prob=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv, no need to use large kernel_size
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.grn = GRN(4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.transpose(1,2)  # (B, C, L) -> (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.grn(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1,2)  # (B, L, C) -> (B, C, L)
        x = shortcut + self.drop_path(x)
        return x

# 1D ConvNeXtV2 + Transformer model for 12-lead ECG
class ConvNeXtV2_1D_ECG(nn.Module):
    def __init__(self, input_channels=12, num_classes=2):
        super().__init__()
        dims = [32, 64, 128, 256]  # Small dimensions
        stages = [2, 2, 6, 2]       # Small block counts
        drop_path_rate = 0.1

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=17, stride=2, padding=3), # try kernel_size= 7, 3, 5, 17, 21, 31
            nn.BatchNorm1d(dims[0]),
            nn.GELU()
        )

        self.blocks = nn.ModuleList()
        dp_rates = [drop_path_rate * (i / (sum(stages) - 1)) for i in range(sum(stages))]
        cur = 0

        for i, num_blocks in enumerate(stages):
            stage = []
            for j in range(num_blocks):
                stage.append(ConvNeXtV2Block1D(dims[i], drop_prob=dp_rates[cur]))
                cur += 1
            self.blocks.append(nn.Sequential(*stage))

            if i < len(stages) - 1:
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    nn.BatchNorm1d(dims[i+1])
                ))

        self.norm = nn.LayerNorm(dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Global Attention Transformer Layer
        self.global_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dims[-1], nhead=4, dim_feedforward=dims[-1]*2, dropout=0.1,        
                batch_first=True,            # operates on [Batch, Seq, Feature]
            ),
            num_layers=1,
            enable_nested_tensor=True    # lets PyTorch pack padded inputs efficiently
        )

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)

        x = x.transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2)
        pooled = self.pool(x).squeeze(-1)

        # Global attention
        global_feat = self.global_attention(x.transpose(1, 2))  # (B, L, C)
        global_feat = global_feat.mean(dim=1)  # Mean pooling after attention

        x = pooled + global_feat  # Merge pooled and global features

        x = self.dropout(x)
        x = self.head(x)
        return x
