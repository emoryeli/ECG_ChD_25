#!/usr/bin/env python

import joblib
import numpy as np
import os
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.signal import resample_poly  # for resampling ECG signals to 400Hz
from math import gcd # for computing upsample and downsample factors for resampling ECG signals
import scipy.signal as sgn
import pywt # wavelet filter for baseline removal
import matplotlib.pyplot as plt
from datetime import datetime

from helper_code import *

# Select device (GPU or MPS or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()  # helpful after large model or dataset loads
elif device.type == 'cpu':
    torch.set_num_threads(os.cpu_count())  # or 16

THRESHOLD_PROBABILITY = 0.73 # above this threshold, the model predicts Chagas disease
source_string = '# Source:' # used to get the source of the data in the traiing set: CODE-15%, PTB-XL, or SaMi-Trop
SAMI = 'SaMi-Trop' # used to limit SaMi data size in the training set
PTBXL = 'PTB-XL' # used to limit PTB-XL data size in  the training set
CODE15 = 'CODE-15%' # used to limit CODE-15% data size in the traiing set 
ECG_len = 4096 #  4096 or 5000 or 2934
EPOCHS = 20 # number of epochs to train the model

# encode sex: M=1, F=0
def encode_sex(sex_str):
    return 1.0 if str(sex_str).lower().startswith('m') else 0.0

class ECGDataset(Dataset):
    def __init__(self, records, labels, smoothing_flags, ages, sexes):
        self.records = records
        self.labels = labels
        self.smoothing_flags = smoothing_flags
        self.ages = ages
        self.sexes = sexes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal = extract_ECG(self.records[idx])
        #assert not torch.isnan(signal).any(), "NaN in input signal!"
        #assert not torch.isinf(signal).any(), "Inf in input signal!"    
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        smooth = torch.tensor(self.smoothing_flags[idx], dtype=torch.bool)
        # Age and sex: normalize age (for stability), encode sex: M=1, F=0
        age = torch.tensor(self.ages[idx] / 100.0, dtype=torch.float32)
        sex = torch.tensor(self.sexes[idx], dtype=torch.float32)
        meta = torch.stack([age, sex])  # shape (2,)
        return signal, label, smooth, meta

# Generalized Cross Entropy loss (GCE) with label smoothing and class weighting
def gce_loss(outputs, labels, smoothing_flags, weight, q=0.7, smoothing=0.1):
    """
    Generalized Cross Entropy loss.
    outputs: logits, shape (batch, num_classes)
    labels: class indices, shape (batch,)
    smoothing_flags: (batch,) True for label-smoothing (CODE-15% data)
    weight: (num_classes,) class weights, e.g., torch.tensor([1.0, 19.0])
    q: parameter for GCE (0 < q <= 1). If q=1, reduces to cross-entropy.
    smoothing: label smoothing value for weak labels.
    """
    outputs = torch.clamp(outputs, min=-20, max=20)
    num_classes = outputs.shape[1]
    device = outputs.device

    # One-hot encode labels with optional smoothing
    with torch.no_grad():
        true_dist = torch.zeros_like(outputs)
        true_dist.scatter_(1, labels.unsqueeze(1), 1)
        if smoothing > 0:
            mask = smoothing_flags.unsqueeze(1)
            true_dist = torch.where(
                mask,
                true_dist * (1 - smoothing) + smoothing / num_classes,
                true_dist
            )

    # Convert logits to probabilities
    probs = F.softmax(outputs, dim=1)
    # GCE formula: L = (1 - p_t^q) / q
    # where p_t is the predicted prob for the true class

    # Multiply by class weights
    class_weights = weight.to(device).unsqueeze(0)  # (1, num_classes)
    p_t = (probs * true_dist).sum(dim=1)  # (batch,)
    w_t = (class_weights * true_dist).sum(dim=1)  # (batch,)

    loss = (1 - p_t ** q) / q
    loss = loss * w_t
    loss = loss.mean()
    return loss

def custom_focal_loss(outputs, labels, weight, gamma=1.0):
    """
    Custom Focal Loss (with class weights and label smoothing).
    outputs: logits (batch, num_classes)
    labels: class indices (batch,)
    weight: (num_classes,) class weights, e.g., torch.tensor([1.0, 19.0])
    gamma: focusing parameter
    """
    outputs = torch.clamp(outputs, min=-20, max=20)
    device = outputs.device

    # One-hot encode labels with optional smoothing
    with torch.no_grad():
        true_dist = torch.zeros_like(outputs)
        true_dist.scatter_(1, labels.unsqueeze(1), 1)

    log_probs = F.log_softmax(outputs, dim=1)
    probs = log_probs.exp()
    focal_term = (1.0 - probs) ** gamma
    class_weights = weight.to(device).unsqueeze(0)
    loss = - (true_dist * focal_term * log_probs * class_weights).sum(dim=1).mean()
    return loss

def combined_focal_gce_loss(
    outputs, labels, smoothing_flags, weight, 
    gamma=1.0, smoothing=0.1, 
    gce_q=0.8,
    lambda_focal=0.5, lambda_gce=0.5
):
    """
    Weighted sum of Custom Focal loss and GCE loss.
    """
    loss_focal = custom_focal_loss(outputs, labels, weight, gamma=gamma)
    loss_gce = gce_loss(outputs, labels, smoothing_flags, weight, q = gce_q, smoothing=smoothing)
    total_loss = lambda_focal * loss_focal + lambda_gce * loss_gce
    return total_loss

# Softmax (sigmoid for binary) approximated top-k true positive rate loss that is differentiable
def soft_topk_tpr_loss(outputs, labels, k_frac=0.05, temperature=0.05):
    """
    outputs: (batch,) - logits or scores, higher = more positive
    targets: (batch,) - binary ground truth (0 or 1)
    k_frac: fraction of batch to be considered "top k" (e.g., 0.05 for top 5%)
    temperature: controls sharpness, lower = harder selection
    """
    #Clamp outputs to avoid Inf
    outputs = torch.clamp(outputs, min=-20, max=20)

    batch_size = outputs.shape[0]
    k = int(k_frac * batch_size)
    if k < 1: k = 1

    # Get softmax weights (sharpen with low temperature)
    soft_weights = F.softmax(outputs / temperature, dim=0)  # shape (batch,)

    # Softly pick top-k: scale so the sum of selected weights is k
    soft_topk_weights = soft_weights * (batch_size / k)
    soft_topk_weights = torch.clamp(soft_topk_weights, max=1.0)  # avoid "overshooting"

    # Expected true positives in top-k
    tps = (soft_topk_weights * labels).sum()
    total_positives = labels.sum().clamp(min=1)

    tpr_at_k = tps / total_positives
    loss = 1 - tpr_at_k
    return loss

def joint_loss(
    outputs, labels, 
    smoothing_flags, # bool tensor for label smoothing, shape (batch,), True only for CODE-15% data 
    weight, # tensor([w_neg, w_pos]) for class weights
    loss_type="combined",  # "custom_focal", "gce", or "combined"
    gamma=1.0, # focusing parameter for focal loss 0.5-2.0 is typical
    smoothing=0.1, # label smoothing for weakly labeled CODE-15% data
    gce_q=0.8, # GCE parameter (0 < q <= 1), lower = more noise-robust, 1 = standard cross-entropy
    lambda_focal=0.5, # fraction of focal loss in the total loss
    lambda_gce=0.5, # fraction of SCE loss in the total loss
    k_frac=0.05,   # fraction of batch to be considered "top k"
    temperature=0.05, # temperature for soft_topk_tpr_loss
    alpha=0.3  # tradeoff parameter for joint loss: alpha=0 means only SCE loss, alpha=1 means only TPR@5% loss
):
    """
    alpha: tradeoff parameter, 0=only main loss, 1=only soft_topk_tpr_loss
    loss_type: choose between "custom_focal", "gce", or "combined"
    """
    #outputs = torch.clamp(outputs, min=-20, max=20)

    # 1. Main loss
    if loss_type == "custom_focal":
        main_loss = custom_focal_loss(outputs, labels, smoothing_flags, weight, gamma=gamma, smoothing=smoothing)
    elif loss_type == "gce":
        main_loss = gce_loss(outputs, labels, smoothing_flags, weight, smoothing=smoothing, q=gce_q)
    elif loss_type == "combined":
        main_loss = combined_focal_gce_loss(
            outputs, labels, smoothing_flags, weight, 
            gamma=gamma, smoothing=smoothing, 
            gce_q=gce_q,
            lambda_focal=lambda_focal, lambda_gce=lambda_gce)
    else:
        raise ValueError("Unknown loss_type: choose 'custom_focal', 'sce', or 'combined'.")

    # 2. Ranking loss on positive class only (class 1)
    pos_outputs = outputs[:, 1]          # logits for class 1
    pos_labels = (labels == 1).float()   # binary 0/1
    loss_tpr = soft_topk_tpr_loss(
        outputs=pos_outputs,
        labels=pos_labels,
        k_frac=k_frac,
        temperature=temperature
    )
    # 3. Weighted sum
    total_loss = (1 - alpha) * main_loss + alpha * loss_tpr
    return total_loss 

def compute_f1_and_thres(labels, probs, steps=100):
    """
    Args:
        labels: 1D numpy array, ground truth (0/1)
        probs: 1D numpy array, predicted probabilities (float)
        steps: number of threshold candidates to try (default: 100)
    Returns:
        best_threshold: threshold (float) that gives highest F1
        best_f1: F1-score at that threshold
    """
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.linspace(0.0, 1.0, steps+1)
    for th in thresholds:
        preds = (probs >= th).astype(int)
        f1 = compute_f_measure(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
    return best_threshold, best_f1

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the training data...')

    # Find the records and add maximum MAX_CODE15 CODE-15% and max MAX_TOTAL total data from the training set 
    # SaMi:CODE15%:PTB-XL:Total  1000:0:19000:20000; 1000:1000:18400:20400; 1000:5000:16000:22000; 1000:10000:13000:24000; 1000:15000:10000:26000; 1000:16000:19000:36000
    # SaMi:CODE15%:PTB-XL:Total  1200:3000:21000:25200; 1631:19169:19500:40300 (might be too much data for training with 16vCPUs on AWS)
    MAX_SAMI = 1631 # all positives
    MAX_CODE15 = 13369 # ~2% positives but weakly labeled (patient self-reported, not confirmed by a serological test)
    MAX_PTBXL = 21000 # all negatives from Europe not from the endemic region South America
    MAX_TOTAL = 36000 # this should give Chagas prevalence <= 5% in the training set, hopefully this can finish training in 72 hours on 16vCPUs on aws

    all_records = find_records(data_folder)
    records = []
    sami_count = 0
    ptbxl_count = 0
    code15_count = 0
    label_smoothing_flags = []

    for record in all_records:
        if len(records) >= MAX_TOTAL:
            break
        header = load_header(os.path.join(data_folder, record))
        source, has_source = get_variable(header, source_string)

        if source == SAMI:
            if sami_count < MAX_SAMI:
                records.append(record)
                label_smoothing_flags.append(False)
                sami_count += 1
            # else: skip, as we've already added maximum SaMi-Trop records
        elif source == PTBXL:
            if ptbxl_count < MAX_PTBXL:
                records.append(record)
                label_smoothing_flags.append(False)
                ptbxl_count += 1
            # else: skip, as we've already added maximum PTB-XL records
        elif source == CODE15:
            if code15_count < MAX_CODE15:
                records.append(record)
                label_smoothing_flags.append(True) # label smoothing only on CODE-15% data
                code15_count += 1
            # else: skip, as we've already added maximum CODE-15% records
        #else:
        #    records.append(record)

    if len(records) == 0:
        raise FileNotFoundError('No useful data were provided.')

    if verbose:
        print('Extracting labels...')

    labels = [load_label(os.path.join(data_folder, rec)) for rec in records]
    ages = [float(load_age(os.path.join(data_folder, rec))) for rec in records]
    sexes = [encode_sex(load_sex(os.path.join(data_folder, rec))) for rec in records]

    data_paths = [os.path.join(data_folder, r) for r in records]
    labels = np.array(labels)
    label_smoothing_flags = np.array(label_smoothing_flags)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    best_val_loss_overall = float('inf')
    best_val_challenge_score_overall = 0.0
    best_model_state_overall = None

    # Store training/validation loss for plotting
    #all_train_losses = []
    #all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels), 1):
        if verbose:
            print(f"\nStarting Fold {fold}...")

        train_records = [data_paths[i] for i in train_idx]
        val_records = [data_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_smoothing_flags = label_smoothing_flags[train_idx]
        val_smoothing_flags = label_smoothing_flags[val_idx]  # not actually needed for validation

        #train_dataset = ECGDataset(train_records, train_labels, train_smoothing_flags)
        #val_dataset = ECGDataset(val_records, val_labels, val_smoothing_flags)
        train_dataset = ECGDataset(train_records, train_labels, train_smoothing_flags, [ages[i] for i in train_idx], [sexes[i] for i in train_idx])
        val_dataset = ECGDataset(val_records, val_labels, val_smoothing_flags, [ages[i] for i in val_idx], [sexes[i] for i in val_idx])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

        model = ConvNeXtV2_1D_ECG().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
        weight = torch.tensor([1.0, 9.5])  # assuming 5% positive -> 1:19 imbalance

        best_val_loss = float('inf')
        best_val_challenge_score = 0.0

        #train_losses = []
        #val_losses = []

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(EPOCHS): 
            model.train()
            total_loss = 0
            for X, y, smoothing_flags, meta in train_loader:
                optimizer.zero_grad()
                outputs = model(X.to(device), meta.to(device)).contiguous()  # logits from model, shape (batch, 2)
                #if torch.isnan(outputs).any():
                #    print("NaN in model output during training!")
                loss = joint_loss(
                    outputs, y.to(device), 
                    smoothing_flags.to(device), # bool tensor for label smoothing, shape (batch,), True only for CODE-15% data 
                    weight, # tensor([w_neg, w_pos]) for class weights
                    loss_type="combined",  # "custom_focal", "gce", or "combined"
                    gamma=1.0, # focusing parameter for focal loss 0.5-2.0 is typical
                    smoothing=0.1, # label smoothing for weakly labeled CODE-15% data
                    gce_q=0.8, # GCE parameter (0 < q <= 1), lower = more noise-robust
                    lambda_focal=0.5, # fraction of focal loss in the total loss
                    lambda_gce=0.5, # fraction of SCE loss in the total loss
                    k_frac=0.05,   # fraction of batch to be considered "top k"
                    temperature=0.05, # temperature for soft_topk_tpr_loss
                    alpha=0.3  # tradeoff parameter for joint loss: alpha=0 means only SCE loss, alpha=1 means only TPR@5% loss
                )
                #if torch.isnan(loss):
                #        print("NaN in training loss!")  
                #loss = custom_loss(outputs, y.to(device), smooth_flag.to(device), weight)
                loss.backward()
                total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # gradient clipping (for long ECGs)
                # check total gradient norm to ensure no exploding gradients:
                """ total_norm_after = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_after += param_norm.item() ** 2
                total_norm_after = total_norm_after ** 0.5
                print(f"Grad Norm before: {total_norm_before:.3f}, after clipping: {total_norm_after:.3f}") """

                optimizer.step()
                total_loss += loss.item() * X.size(0) 
            avg_train_loss = total_loss / len(train_loader.dataset)
            #train_losses.append(avg_train_loss)  # track train loss for plotting

            model.eval()
            val_loss = 0
            val_outputs = [] #predicted probabilities, float64
            val_targets = [] #labels, 0 or 1
            
            #validation loop
            with torch.no_grad():
                for X, y, smoothing_flags, meta in val_loader:
                    outputs = model(X.to(device), meta.to(device))
                    loss = joint_loss(
                        outputs, y.to(device), 
                        smoothing_flags.to(device), # bool tensor for label smoothing, shape (batch,), True only for CODE-15% data 
                        weight, # tensor([w_neg, w_pos]) for class weights
                        loss_type="combined",  # "custom_focal", "gce", or "combined"
                        gamma=1.0, # focusing parameter for focal loss 0.5-2.0 is typical
                        smoothing=0.1, # label smoothing for weakly labeled CODE-15% data
                        gce_q=0.8, # GCE parameter (0 < q <= 1), lower = more noise-robust
                        lambda_focal=0.5, # fraction of focal loss in the total loss
                        lambda_gce=0.5, # fraction of SCE loss in the total loss
                        k_frac=0.05,   # fraction of batch to be considered "top k"
                        temperature=0.05, # temperature for soft_topk_tpr_loss
                        alpha=0.3  # tradeoff parameter for joint loss: alpha=0 means only SCE loss, alpha=1 means only TPR@5% loss
                    )
                    #if torch.isnan(loss):
                    #    print("NaN in validation loss!")
                    val_loss += loss.item() * X.size(0)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # probs for class 1: Chagas
                    val_outputs.extend(probs.tolist())
                    val_targets.extend(y.cpu().numpy().tolist())
            avg_val_loss = val_loss / len(val_loader.dataset)
            #val_losses.append(avg_val_loss)  # track val loss for plotting

            scheduler.step(avg_val_loss)
            #f1 = compute_f_measure(val_targets, (np.array(val_outputs) > THRESHOLD_PROBABILITY).astype(int))
            #best_threshold, best_f1 = compute_f1_and_thres(val_labels, np.array(val_outputs))
            challenge_score = compute_challenge_score(np.array(val_labels), np.array(val_outputs))
            #auroc, auprc = compute_auc(np.array(val_labels), np.array(val_outputs))

            gc.collect()  # free memory after each epoch
            
            #if verbose:
            #    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            #    print(f"Fold {fold}, Epoch {epoch + 1}: "
            #        f"Train Loss = {avg_train_loss:.3f}, "
            #        f"Val Loss = {avg_val_loss:.3f}, "
                    #f"F1_0p5 = {f1:.3f}, "
            #        f"F1_Best = {best_f1:.3f}, "
            #        f"Thres_Best = {best_threshold:.3f}, "
            #        f"AUROC = {auroc:.3f}, "
            #        f"AUPRC = {auprc:.3f}, "
            #        f"Challenge Score = {challenge_score:.3f}, "
            #        f"LR = {optimizer.param_groups[0]['lr']:.2e}")
                
                # Check how many true positives are in top 5%
                #val_targets = np.array(val_targets)
                #val_outputs = np.array(val_outputs)

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

            #if avg_val_loss < best_val_loss:
            if challenge_score > best_val_challenge_score: # use challenge score instead of validation loss
                #best_val_loss = avg_val_loss
                best_val_challenge_score = challenge_score
                best_model_state = model.state_dict()

            os.makedirs(model_folder, exist_ok=True)
            torch.save({'model_state_dict': best_model_state}, os.path.join(model_folder, 'model.pt'))

        #all_train_losses.append(train_losses) # training loss for plotting
        #all_val_losses.append(val_losses) # validation loss for plotting

        #if best_val_loss < best_val_loss_overall:
        if best_val_challenge_score > best_val_challenge_score_overall: # use challenge score instead of validation loss
            #best_val_loss_overall = best_val_loss
            best_val_challenge_score_overall = best_val_challenge_score
            best_model_state_overall = best_model_state

    #os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': best_model_state_overall}, os.path.join(model_folder, 'model.pt'))

    """ if verbose:
        # --- Plot Loss Curves ---
        plt.figure(figsize=(10, 6))
        for fold, (train_l, val_l) in enumerate(zip(all_train_losses, all_val_losses), 1):
            plt.plot(train_l, label=f'Fold {fold} Train')
            plt.plot(val_l, '--', label=f'Fold {fold} Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss Curves (All Folds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"\nBest model saved with validation loss: {best_val_loss_overall:.4f}") """

def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_folder, 'model.pt'))

def load_model(model_folder, verbose):
    model = ConvNeXtV2_1D_ECG()
    checkpoint = torch.load(os.path.join(model_folder, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_model(record, model, verbose):
    x = extract_ECG(record).unsqueeze(0)
    age = torch.tensor(float(load_age(record)) / 100.0, dtype=torch.float32).unsqueeze(0)
    sex = torch.tensor(encode_sex(load_sex(record)), dtype=torch.float32).unsqueeze(0)
    meta = torch.cat([age, sex], dim=0).unsqueeze(0)  # shape (1,2)
    output = model(x, meta)
    probs = torch.softmax(output, dim=1).detach().numpy()[0]
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
    # ECG_len = 4096 or 5000 or 2934, some SaMi-Trop and CODE-15% data have 2934 samples
    if signal.shape[1] < ECG_len:
        pad_width = ECG_len - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)))
    else: # random cropping for data augmentation
        signal = random_crop(signal, target_length=ECG_len)

    signal = ECG_preprocess(signal, sample_rate=target_fs, powerline_freqs=[60, 50], bandwidth=[0.05, 150], augment=True, target_length=ECG_len)

    return torch.tensor(signal.copy(), dtype=torch.float32)

# ECG pre-processing functions:
# some adapted from https://github.com/antonior92/ecg-preprocessing/
""" def remove_baseline_butterworth_filter(sample_rate=400):
    # Butterworth highpass filter design, preferred to elliptic filters for ECG baseline wanter removal
    fc = 0.5  # [Hz], cutoff frequency, 0.5-0.7 Hz, never above 0.8Hz which will distort T wave and ST segment
    order = 3  # Butterworth order (standard is 2-4 for ECG)
    nyquist = 0.5 * sample_rate
    wn = fc / nyquist  # Normalized frequency (0-1)
    sos = sgn.butter(order, wn, btype='highpass', output='sos')
    return sos """

def remove_baseline_wavelet_filter(ecg_12lead, wavelet='sym8', level=7):
    """
    Baseline removal for 12-lead ECG with shape (12, length)
    ecg_12lead: 12 lead ECG (12, length)
    wavelet:  Wavelet type (e.g., 'sym8' or Daubechies 6 'db6' is commonly used for ECG)
    level:  Decomposition level (higher = slower baseline removed, level 8 for 400hz sampling rate removes frequencies slower than ~0.78 Hz)
    level 9 (0.039hz) is too high for 4096 length
    Returns: cleaned ECG, same shape (12, length)
    """
    leads, length = ecg_12lead.shape
    cleaned_ecg = np.zeros_like(ecg_12lead)
    for lead in range(leads):
        coeffs = pywt.wavedec(ecg_12lead[lead], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation (baseline)
        cleaned_lead = pywt.waverec(coeffs, wavelet)
        cleaned_ecg[lead] = cleaned_lead[:length]  # In case waverec pads
    return cleaned_ecg

def remove_powerline_filter(powerline_freq=60, sample_rate=400): # 60 Hz for US, Brazil, 50 Hz for Europe, China
    # Design notch filter
    q = 30.0  # Quality factor
    b, a = sgn.iirnotch(powerline_freq, q, fs=sample_rate)
    return b, a 

def bandpass_filter(bandwidth=[0.5, 47], sample_rate=400):
    # --- Bandpass filtering with Butterworth bandpass filter (zero-phase) ---
    # 3rd order Butterworth, passband: [0.5, 47] Hz
    nyquist = 0.5 * sample_rate
    low = bandwidth[0] / nyquist
    high = bandwidth[1] / nyquist
    sos_band = sgn.butter(3, [low, high], btype='bandpass', output='sos')
    return sos_band

def normalize_signal(signal):
    # Normalize the signal - z-score normalization
    x = signal
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    # Avoid division by zero or NaN
    std[(std == 0) | np.isnan(std)] = 1
    x = (x - mean) / std
    return x

# ECG data augmentation functions:
def random_crop(signal, target_length=ECG_len):
    # signal shape: (leads, samples)
    num_samples = signal.shape[-1]
    if num_samples <= target_length:
        # Pad if too short
        pad_width = target_length - num_samples
        signal = np.pad(signal, ((0,0), (0, pad_width)))
        return signal
    else:
        start = np.random.randint(0, num_samples - target_length + 1)
        return signal[:, start:start+target_length]

def add_gaussian_noise(signal, noise_std=0.02):
    # noise_std can be tuned; 0.01–0.05 is typical
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise

def random_amplitude_scaling(signal, min_scale=0.8, max_scale=1.2):
    scale = np.random.uniform(min_scale, max_scale)
    return signal * scale

def random_time_shift(signal, max_shift=50):
    shift = np.random.randint(-max_shift, max_shift+1)
    return np.roll(signal, shift, axis=-1)

def random_lead_dropout(signal, dropout_prob=0.1):
    # Randomly zero out one lead
    if np.random.rand() < dropout_prob:
        lead = np.random.randint(0, signal.shape[0])
        signal = signal.copy()
        signal[lead, :] = 0
    return signal

def add_baseline_wander(signal, sample_rate=400, max_ampl=0.1, freq_range=(0.15, 0.3)):
    # Adds low-frequency baseline wander
    t = np.arange(signal.shape[1]) / sample_rate
    amp = np.random.uniform(0, max_ampl)
    freq = np.random.uniform(*freq_range)
    phase = np.random.uniform(0, 2 * np.pi)
    drift = amp * np.sin(2 * np.pi * freq * t + phase)
    return signal + drift

def ECG_preprocess(signal, sample_rate=400, powerline_freqs=[60, 50], bandwidth=[0.5, 47], augment=False, target_length=ECG_len):
    # signal shape: (samples,) or (channels, samples)
    x = signal

    # Reflection padding on each side of the signal to avoid edge effects in filtering
    pad_sec = 1  # seconds to pad on each side
    pad_width = int(pad_sec * sample_rate)
    # Reflection padding along time axis (axis=1)
    x = np.pad(x, ((0,0),(pad_width, pad_width)), mode='reflect')

    # Remove baseline drift (Butterworth highpass filter)
    #sos = remove_baseline_butterworth_filter(sample_rate)
    #x = sgn.sosfiltfilt(sos, x, axis=-1) # , padtype='constant'

    # Remove baseline drift using wavelet filter
    x = remove_baseline_wavelet_filter(x)

    # Remove DC baseline shift (mean)
    x = x - np.nanmean(x, axis=-1, keepdims=True)
    
    # Remove powerline interference (notch filters)
    for freq in powerline_freqs:
        b, a = remove_powerline_filter(freq, sample_rate)
        x = sgn.filtfilt(b, a, x, axis=-1)

    # Bandpass filter the signal to bandwidth [low, high] Hz
    sos_band = bandpass_filter(bandwidth, sample_rate)
    x = sgn.sosfiltfilt(sos_band, x, axis=-1)

    # Remove padding
    x = x[:, pad_width:-pad_width]  

    # Remove edge artifacts in the first and last 0.25 seconds 
    # (first 100 and last 100 samples, since 400 Hz × 0.25 s = 100)
    # replace with the mean of the lead:
    #x[:, :100] = x[:, -100:] = np.nanmean(x, axis=1, keepdims=True)
    # replace with zeroes:
    x[:, :100] = x[:, -100:] = 0.0
    
    # Normalize the signal - z-score normalization
    x = normalize_signal(x)

    # ECG signal augementation:
    if augment:
        # 1. Random crop to target length
        # already done in extract_ECG()
        # x = random_crop(x, target_length=target_length)
        # 2. Random amplitude scaling
        if np.random.rand() < 0.5:
            x = random_amplitude_scaling(x)
        # 3. Add Gaussian noise
        if np.random.rand() < 0.5:
            x = add_gaussian_noise(x)
        # 4. Time shift
        if np.random.rand() < 0.5:
            x = random_time_shift(x)
        # 5. Lead dropout
        if np.random.rand() < 0.2:
            x = random_lead_dropout(x)
        # 6. Optional: add baseline wander
        # if np.random.rand() < 0.2:
        #     x = add_baseline_wander(x, sample_rate)
    return x

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
    return x.div(keep_prob) * random_tensor # rescales the output to preserve the expected value

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
    
# 1D ConvNeXt V2 basic block
class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, dim, drop_prob=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv, no need to use large kernel_size
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # inverted bottleneck: 1x1 conv
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
    
# 1D ConvNeXtV2 model: ~10.34 million parameter for [64, 128, 256, 512]; ~ 2.37 million parameters for [32, 64, 128, 256]
class ConvNeXtV2_1D_ECG(nn.Module):
    def __init__(self, input_channels=12, num_classes=2):
        super().__init__()
        dims = [64, 128, 256, 512]  # Small dimensions [32, 64, 128, 256], try larger dims like [64, 128, 256, 512] if data and compute allow
        stages = [2, 2, 6, 2]       # Small block counts, try larger stages like [3, 3, 9, 3] if data and compute allow
        drop_path_rate = 0.1

        self.stem = nn.Sequential(
            # non-overlapping convolution: stride = kernel_size ('patchify' like ViT), tried kernel_size = 7, 3, 5, 17, 21, 33, 65: 17 is the best
            nn.Conv1d(input_channels, dims[0], kernel_size=17, stride=17),
            #nn.BatchNorm1d(dims[0]),
            nn.GroupNorm(8, dims[0]),
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
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2), # downsampling 
                    #nn.BatchNorm1d(dims[i+1])
                    nn.GroupNorm(8, dims[i+1]) 
                ))

        self.norm = nn.LayerNorm(dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(dims[-1] + 2, 256), # +2 for age, sex
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, meta):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)

        x = x.transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2)  # (B, C, L) -> (B, L, C) -> (B, C, L)
        x = self.pool(x).squeeze(-1)  # (B, C, 1) -> (B, C)

        x = self.dropout(x)
        x = torch.cat([x, meta], dim=1)  # concatenate meta features: age, sex
        x = self.head(x)
        return x
