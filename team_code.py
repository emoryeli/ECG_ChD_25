#!/usr/bin/env python

import joblib
import numpy as np
import os
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from scipy.signal import resample_poly  # for resampling ECG signals to 400Hz
from math import gcd
import scipy.signal as sgn
import pywt # wavelet filter for baseline removal
import copy
#import matplotlib.pyplot as plt
from datetime import datetime

from helper_code import *

# Select device (GPU or MPS or CPU): 16 vCPUs 64GB RAM on AWS
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()  # helpful after large model or dataset loads
elif device.type == 'cpu':
    torch.set_num_threads(os.cpu_count())  # or 16

THRESHOLD_PROBABILITY = 0.86 # above this threshold, the model predicts Chagas disease
source_string = '# Source:' # used to get the source of the data in the traiing set: CODE-15%, PTB-XL, or SaMi-Trop
SAMI = 'SaMi-Trop' # used to limit SaMi data size in the training set
PTBXL = 'PTB-XL' # used to limit PTB-XL data size in  the training set
CODE15 = 'CODE-15%' # used to limit CODE-15% data size in the traiing set 
POSITIVE_RATIO = 0.5 # positive ratio in the training set, used for over sampling positives
ECG_len = 4096 #  4096 or 5000 or 2934: double-check: all Chagas positives in CODE-15% are 2934 long
EPOCHS = 12 # number of epochs to train the model

class ECGDataset(Dataset):
    def __init__(self, records, labels, smoothing_flags, augment=False):
        self.records = records
        self.labels = labels
        self.smoothing_flags = smoothing_flags
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal = extract_ECG(self.records[idx], augment=self.augment)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        smooth = torch.tensor(self.smoothing_flags[idx], dtype=torch.bool)
        return signal, label, smooth

# custom Focal loss function with weighted label and label smoothing on CODE-15% data 
def custom_focal_loss(outputs, targets, smoothing_flags, weight, smoothing=0.02, gamma=2.0, reduction='mean'):
    # outputs: logits (batch, num_classes)
    # targets: class indices (batch,)
    # smoothing_flags: (batch,)
    # weight: (num_classes,) class weights, e.g., torch.tensor([1.0, 19.0])
    # gamma: focusing parameter
    # smoothing: label smoothing value

    #Clamp outputs to avoid NaN/Inf
    #outputs = torch.clamp(outputs, min=-20, max=20)

    num_classes = outputs.shape[1]
    device = outputs.device

    # Label smoothing
    with torch.no_grad():
        true_dist = torch.zeros_like(outputs)# (batch, num_classes = 2)
        true_dist.scatter_(1, targets.unsqueeze(1), 1)
        mask = smoothing_flags.unsqueeze(1)  # (batch, 1), bool
        true_dist = torch.where(
            mask,
            true_dist * (1 - smoothing) + smoothing / num_classes,
            true_dist
        )

    # Log-softmax and softmax
    log_probs = F.log_softmax(outputs, dim=1)  # (batch, num_classes)
    probs = log_probs.exp()                    # (batch, num_classes)

    # Focal Loss term: (1 - pt) ** gamma
    focal_term = (1.0 - probs) ** gamma # larger gamma focuses more on hard examples

    # Class weights (alpha)
    class_weights = weight.to(device).unsqueeze(0)  # (1, num_classes)

    # Per sample loss - Combine all terms
    per_sample_loss = -(true_dist * focal_term * log_probs * class_weights).sum(dim=1)

    return per_sample_loss if reduction == 'none' else per_sample_loss.mean()

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

# oversample positives to positive_ratio of overall data (not per epoch)
def create_oversampled_dataloader(dataset, labels, batch_size=64, positive_ratio=0.5, num_workers=12):
    labels = np.array(labels)
    pos = (labels == 1)
    neg = (labels == 0)
    n_pos = pos.sum()
    n_neg = neg.sum()
    
    # Set weights so that sampled positives ≈ positive_ratio
    pos_weight = positive_ratio / (1 - positive_ratio) * n_neg / n_pos 
    neg_weight = 1
    #print(pos_weight, neg_weight)
    weights = np.where(labels == 1, pos_weight, neg_weight)
    
    sampler = WeightedRandomSampler(weights, len(labels), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)

class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        """msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            v = msd[k]
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(d).add_(v, alpha=1 - d)  # smooth float tensors
            else:
                esd[k].copy_(v)  # copy ints (e.g., num_batches_tracked)"""
        d_bn = 0.9 # for BatchNorm buffers, use a smaller decay; d_bn = 0 is simple copy
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in msd.items():
            if not esd[k].dtype.is_floating_point:
                esd[k].copy_(v)  # copy ints (e.g., num_batches_tracked)
            # For BatchNorm buffers EMA, use a smaller decay d_bn 
            elif k.endswith('running_mean') or k.endswith('running_var'):
                esd[k].mul_(d_bn).add_(v, alpha=1 - d_bn)
            else:
                esd[k].mul_(d).add_(v, alpha=1 - d)

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the training data...')

    # Find the records SaMi:CODE15%:PTB-XL:Total:
    # 1631:28569:21000:51200; 
    # 1000:31200:19000:51200; 
    MAX_SAMI = 1631 # all positives
    MAX_CODE15 = 28569 # ~1.91% positives but weakly labeled (patient self-reported, not confirmed by a serological test)
    MAX_PTBXL = 21000 # all negatives from Germany not from the endemic region South America
    MAX_TOTAL = 51200 # mutiples of batch_size to avoid last batch size mismatch; this should give Chagas prevalence <= 5% in the training set, hopefully this can finish training in 72 hours on 16vCPUs on aws

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
        label = load_label(os.path.join(data_folder, record))
        source, has_source = get_variable(header, source_string)

        if source == SAMI and sami_count < MAX_SAMI:
                records.append(record)
                label_smoothing_flags.append(False)
                sami_count += 1
            # else: skip, as we've already added maximum SaMi-Trop records
        elif source == PTBXL and ptbxl_count < MAX_PTBXL:
                records.append(record)
                label_smoothing_flags.append(False)
                ptbxl_count += 1
            # else: skip, as we've already added maximum PTB-XL records
        elif source == CODE15 and label == 0 and code15_count < MAX_CODE15:
                records.append(record)
                label_smoothing_flags.append(True) # label smoothing only on CODE-15% data
                code15_count += 1
            # else: skip, as we've already added maximum CODE-15% records
        #else:
        #    records.append(record)
    #print(sami_count, ptbxl_count, code15_count)

    if len(records) == 0:
        raise FileNotFoundError('No useful data were provided.')

    if verbose:
        print('Extracting labels...')

    labels = [load_label(os.path.join(data_folder, rec)) for rec in records]
    data_paths = [os.path.join(data_folder, r) for r in records]
    labels = np.array(labels)
    label_smoothing_flags = np.array(label_smoothing_flags)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 12345, 8011
    #best_val_loss_overall = float('inf')
    best_challenge_score_overall = -1.0
    best_model_state_overall = None

    os.makedirs(model_folder, exist_ok=True) # save the best model for each fold as model_{fold}.pt, and the best model overall as model.pt
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels), 1):
        if verbose:
            print(f"\nStarting Fold {fold}...")

        train_records = [data_paths[i] for i in train_idx]
        val_records = [data_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_smoothing_flags = label_smoothing_flags[train_idx]
        val_smoothing_flags = label_smoothing_flags[val_idx] 

        train_dataset = ECGDataset(train_records, train_labels, train_smoothing_flags, augment=True)
        val_dataset = ECGDataset(val_records, val_labels, val_smoothing_flags, augment=False)
        #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
        train_loader = create_oversampled_dataloader(train_dataset, train_labels, batch_size=64, positive_ratio=POSITIVE_RATIO, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        model = ConvNeXtV2_1D_ECG().to(device)
        # print(f"# of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}" )
        ema = ModelEMA(model, decay=0.999, device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-6)

        weight = torch.tensor([1.0, 9.0]) # (1-POSITIVE_RATIO)/POSITIVE_RATIO

        #best_val_loss = float('inf')
        best_challenge_score = -1.0

        #torch.autograd.set_detect_anomaly(True)

        # training loop
        for epoch in range(1, EPOCHS+1): 
            model.train()
            total_loss = 0
            for X, y, smooth_flag in train_loader: # 51200/5/64 = 160 iterations per epoch

                optimizer.zero_grad(set_to_none=True)

                outputs = model(X.to(device)).contiguous()  # logits from model, shape (batch, 2)
                #outputs = torch.clamp(outputs, min=-20, max=20)

                y, smooth_flag = y.to(outputs.device), smooth_flag.to(outputs.device)

                per_sample_loss = custom_focal_loss(
                    outputs,         # logits from model, shape (batch, 2)
                    y,         # labels, int64 tensor, shape (batch,)
                    smooth_flag, # bool tensor, shape (batch,)
                    weight,          # tensor([w_neg, w_pos])
                    smoothing=0.02,  # label smoothing only for weakly labeled CODE-15% data, if too many FP's, decrease smoothing
                    gamma=2.0, # focusing parameter for focal loss
                    reduction='none' # per-sample loss, shape (batch, )
                ) 

                # Down-weight hard negatives: non-chagas(label 0) with high predicted probability for Chagas (p > 0.8) (potential mislabels)
                # future: try Huber loss?
                with torch.no_grad():
                    p_pos = torch.softmax(outputs, dim=1)[:, 1]     # [B]
                    code_neg = (y == 0) & smooth_flag # bool mask for CODE-15% negatives
                    hard_neg = (code_neg & (p_pos > 0.8))   # bool mask for hard negatives among CODE-15% negatives (potential mislabels)
                    #print(hard_neg.sum().item(), 'hard negatives in batch') # only 0 or 1 hard negatives in each batch (ok as mislabels should be less than 4%)
                    scale = torch.ones_like(p_pos) # on device
                    scale[hard_neg] = 0.05 # down-weight hard negatives by 0.2, or 0.1, or 0.05, or 0.01

                # Down-weighted loss: weighted mean
                loss = (per_sample_loss * scale).mean()
                #loss = per_sample_loss.mean()

                # Pairwise Ranking (Hinge) Loss
                # using logits:
                logits = outputs[:, 1] - outputs[:, 0]
                # using probabilities:
                #p = torch.softmax(outputs, dim=1)[:, 1]
                pos = (y == 1)
                neg = (y == 0) # excluding hard negatives in CODE-15% data (& ~(hard_neg)) seems to hurt performance

                if pos.any() and neg.any():
                    # using logits:
                    pos_scores = logits[pos].unsqueeze(1)
                    neg_scores = logits[neg]
                    # using probabilities:
                    #pos_scores = p[pos].unsqueeze(1)
                    #neg_scores = p[neg]

                    q = 0.2 # fraction of negatives to sample for ranking loss
                    k = max(1, int(q * neg_scores.numel()))
                    top_neg, _ = neg_scores.topk(k)
                    
                    margin = 1.0 # logit margin, try 0.3-1.0
                    #margin = 0.05 # probability margin
                    rank_loss = (margin + top_neg.unsqueeze(0) - pos_scores).clamp_min(0).mean()
                    # ramp the weight so it matters after the EMA peak
                    w0, w1 = 0.0, 0.1  # start small, end modest 0.0, 0.1
                    t = epoch / max((EPOCHS-1)//2, 1) # ramp from 0.0 to 0.1 in (EPOCHS-1)//2 epochs then stay at 0.1
                    rank_w = w0 + (w1 - w0) * t
                    #rank_w = 0.1
                    #print(f"Rank Loss: {rank_loss.item():.4f}, Loss: {loss.item():.4f}")        
                    loss = loss + rank_w * rank_loss

                loss.backward()

                # Gradient clipping
                total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                #print(total_norm_before)

                optimizer.step()
                ema.update(model)  # update EMA model

                total_loss += loss.item() * X.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            ema.ema.eval()

            val_loss = 0
            val_outputs = [] #predicted probabilities, float64
            val_targets = [] #labels, 0 or 1
            
            # validation loop
            with torch.inference_mode():
                for X, y, smooth_flag in val_loader:
                    outputs = ema.ema(X.to(device))  # logits from EMA model, shape (batch, 2)
                    #outputs = model(X.to(device))
                    #outputs = torch.clamp(outputs, min=-20, max=20)
                    
                    loss = custom_focal_loss(
                        outputs,         # logits from model, shape (batch, 2)
                        y.to(device),         # int64 tensor, shape (batch,)
                        smooth_flag.to(device), # bool tensor, shape (batch,)
                        weight,          # tensor([w_neg, w_pos])
                        smoothing=0.02,  # label smoothing only for weakly labeled CODE-15% data, if too many FP's, decrease smoothing
                        gamma=2.0, # focusing parameter for focal loss
                    )
                    val_loss += loss.item() * X.size(0)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # probs for class 1: Chagas
                    val_outputs.extend(probs.tolist())
                    val_targets.extend(y.cpu().numpy().tolist())
            avg_val_loss = val_loss / len(val_loader.dataset)

            #scheduler.step(avg_val_loss)
            scheduler.step()  # cosine step once per epoch

            val_outputs_np = np.array(val_outputs)
            val_labels_np = np.array(val_targets)
            best_threshold, best_f1 = compute_f1_and_thres(val_labels_np, val_outputs_np)
            challenge_score = compute_challenge_score(val_labels_np, val_outputs_np)
            auroc, auprc = compute_auc(val_labels_np, val_outputs_np)

            gc.collect()  # free memory after each epoch

            if verbose:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Fold {fold}, Epoch {epoch}: "
                    f"Train Loss = {avg_train_loss:.3f}, "
                    f"Val Loss = {avg_val_loss:.3f}, "
                    f"F1_Best = {best_f1:.3f}, "
                    f"Thres_Best = {best_threshold:.3f}, "
                    f"AUROC = {auroc:.3f}, "
                    f"AUPRC = {auprc:.3f}, "
                    f"Challenge Score = {challenge_score:.3f}, "
                    f"LR = {optimizer.param_groups[0]['lr']:.2e}")
                
            #if avg_val_loss <= best_val_loss:
            #    best_val_loss = avg_val_loss
            if challenge_score >= best_challenge_score:
                best_challenge_score = challenge_score
                #best_model_state = model.state_dict()
                best_model_state = copy.deepcopy(ema.ema.state_dict())
                torch.save({'model_state_dict': best_model_state}, os.path.join(model_folder, f'model_{fold}.pt')) 

        #if best_val_loss <= best_val_loss_overall:
        #    best_val_loss_overall = best_val_loss
        if best_challenge_score >= best_challenge_score_overall:
            best_challenge_score_overall = best_challenge_score
            best_model_state_overall = best_model_state

    torch.save({'model_state_dict': best_model_state_overall}, os.path.join(model_folder, 'model.pt'))

def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_folder, 'model.pt'))

"""def load_model(model_folder, verbose):
    model = ConvNeXtV2_1D_ECG()
    checkpoint = torch.load(os.path.join(model_folder, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_model(record, model, verbose):
    x = extract_ECG(record, augment=False).unsqueeze(0)
    output = model(x)
    probs = torch.softmax(output, dim=1).detach().numpy()[0]
    binary_output = int(probs[1] > THRESHOLD_PROBABILITY)  # 1 for Chagas disease, 0 for no Chagas diseas
    return binary_output, probs[1]"""

# load 5 models from the model_folder
def load_individual_model(path):
    model = ConvNeXtV2_1D_ECG()
    # print(f"# trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}" )
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model

@torch.no_grad()
def ensemble_logits(x, models):
    # average logits is better than averaging probs
    return sum(m(x) for m in models) / len(models)

def load_model(model_folder, verbose):
    model_paths = [os.path.join(model_folder, f"model_{i}.pt") for i in range(1, 6)]
    ckpts = [p for p in model_paths if os.path.isfile(p)]
    if len(ckpts) == 0: 
        print(f"No models found in {model_folder}.")
        return None
    else: 
        models = [load_individual_model(p) for p in ckpts]
        return models

def run_model(record, model, verbose): # model is a list of 5 models
    x = extract_ECG(record, augment=False).unsqueeze(0)
    output_logits = ensemble_logits(x, model)
    probs = torch.softmax(output_logits, dim=1).detach().numpy()[0]
    binary_output = int(probs[1] > THRESHOLD_PROBABILITY)  # probs[1] are probabilites for Chagas disease, probs[0] for non-Chagas diseas
    return binary_output, probs[1]

def extract_ECG(record, augment=False):
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
        g = gcd(int(target_fs), int(fs))
        up = target_fs // g
        down = fs // g
        # Resample each lead separately using resample_poly
        signal = np.array([resample_poly(lead, up, down) for lead in signal])

    # Truncate or pad to fixed length (e.g. max_len = 2934 samples for 7.2s at 400 Hz)
    # ECG_len = 4096 or 5000 or 2934, some SaMi-Trop and CODE-15% data have 2934 samples, all Chagas positves in CODE-15% are 2934 long
    if signal.shape[1] < ECG_len:
        pad_width = ECG_len - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='edge')
    else: # random cropping if augment = True
        signal = random_crop(signal, target_length=ECG_len) if augment else signal[:, :ECG_len] 

	# [0.05, 150] seems to give slightly worse challenge score on hidden validation set than [0.5, 59] (0.292 vs 0.293)
    signal = ECG_preprocess(signal, sample_rate=target_fs, powerline_freqs=[60, 50], bandwidth=[0.5, 59], augment=augment, target_length=ECG_len)  
    #assert not np.isnan(signal).any(), "NaN found after ECG_preprocess"

    return torch.tensor(signal.copy(), dtype=torch.float32)

# ECG pre-processing functions:
# some adapted from https://github.com/antonior92/ecg-preprocessing/
def remove_baseline_wavelet_filter(ecg_12lead, wavelet='sym8', level=8): # or 8
    """
    Baseline removal for 12-lead ECG with shape (12, length)
    ecg_12lead: 12 lead ECG (12, length)
    wavelet:  Wavelet type (e.g., Symlet-8 'sym8' or Daubechies 6 'db6' is commonly used for ECG)
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

def bandpass_filter(bandwidth=[0.05, 150], sample_rate=400):
    # --- Bandpass filtering with Butterworth bandpass filter (zero-phase) ---
    # 4th order Butterworth, passband is specified by bandwidth in Hz [low, high]
    nyquist = 0.5 * sample_rate
    low = bandwidth[0] / nyquist
    high = bandwidth[1] / nyquist
    sos_band = sgn.butter(4, [low, high], btype='bandpass', output='sos')
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

def ECG_preprocess(signal, sample_rate=400, powerline_freqs=[60, 50], bandwidth=[0.05, 150], augment=False, target_length=ECG_len):
    # signal shape: (samples,) or (channels, samples)
    x = signal

    # Reflection padding on each side of the signal to avoid edge effects in filtering
    pad_sec = 3  # seconds to pad on each side
    pad_width = int(pad_sec * sample_rate)
    # Reflection padding along time axis (axis=1)
    x = np.pad(x, ((0,0),(pad_width, pad_width)), mode='reflect')

    # Remove powerline interference (notch filters)
    for freq in powerline_freqs:
        b, a = remove_powerline_filter(freq, sample_rate)
        x = sgn.filtfilt(b, a, x, axis=-1)

    # Remove baseline drift using wavelet filter
    x = remove_baseline_wavelet_filter(x)
    
    # Remove baseline drift (Butterworth highpass filter)
    #sos = remove_baseline_butterworth_filter(sample_rate)
    #x = sgn.sosfiltfilt(sos, x, axis=-1) # , padtype='constant'

    # Bandpass filter the signal to bandwidth [low, high] Hz
    sos_band = bandpass_filter(bandwidth, sample_rate)
    x = sgn.sosfiltfilt(sos_band, x, axis=-1)

    # Remove DC baseline shift (mean)
    # x = x - np.nanmean(x, axis=-1, keepdims=True)

    # Remove padding
    x = x[:, pad_width:-pad_width]  

    # Remove edge artifacts: the first and last 0.25 seconds 
    # (first 100 and last 100 samples, since 400 Hz × 0.25 s = 100)
    # replace with the mean of the lead:
    #x[:, :100] = x[:, -100:] = np.nanmean(x, axis=1, keepdims=True)
    # replace with zeroes:
    x[:, :100] = x[:, -100:] = 0.0
    
    # Normalize the signal - z-score normalization
    x = normalize_signal(x)

    # ECG signal augmentation:
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
        #if np.random.rand() < 0.2:
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

# Global Response Normalization (GRN): after testing, it seems not helpful for 1D ECG data
""" class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1)) # (1, C, 1) for (B, C, L) for propoer braodcasting
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps   = eps
    def forward(self, x):                    # x shape (B, C, L)  after transpose
        gx = torch.norm(x, p=2, dim=1, keepdim=True)  # global L2 norm across sequence per channel
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x """
    
""" def Norm1d(C, groups=16): # GroupNorm for 1D data, similar to LayerNorm but with groups
    g = min(groups, C)                 # cap groups by channels
    while C % g != 0 and g > 1:        # ensure divisibility
        g //= 2
    if g == 1:
        # fallback if channels are tiny → InstanceNorm-like behavior
        return nn.GroupNorm(num_groups=1, num_channels=C, eps=1e-6, affine=True)
    return nn.GroupNorm(num_groups=g, num_channels=C, eps=1e-6, affine=True) """

# 1D ConvNeXt V2 basic block
class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, dim, drop_prob=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # tried larger kernel 7 or 15 (worse) padding=(kernel_size-1)//2, depthwise conv, 
        #self.norm = nn.LayerNorm(dim, eps=1e-6)
        #self.norm = Norm1d(dim)
        self.norm = nn.BatchNorm1d(dim)
        # self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1) # inverted bottleneck: 1x1 conv
        self.act = nn.GELU()
        #self.grn = GRN(4 * dim)
        #self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        #self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
        #                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        #x = x.transpose(1,2)  # (B, C, L) -> (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x) 

        x = self.act(x)
        #x = self.grn(x)
        x = self.pwconv2(x)
        #if self.gamma is not None:
        #    x = self.gamma * x
        #x = x.transpose(1,2)  # (B, L, C) -> (B, C, L)
        x = shortcut + self.drop_path(x)
        return x

# downsampling layer for ConvNeXt V2
class Downsample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        #self.norm = Norm1d(in_channels)
        self.norm = nn.BatchNorm1d(in_channels, eps=1e-6)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        #x = x.transpose(1, 2)  # (B, C, L) -> (B, L, C)
        x = self.norm(x)
        #x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        x = self.conv(x)
        return x

# stem for ConvNeXt V2
class Stem1D(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=3): # tried 4, 7 or 15 (worse)
        super().__init__()
        self.conv = nn.Conv1d(input_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)
        #self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        #self.norm = Norm1d(out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)             # [B, C, L]
        #x = x.transpose(1, 2)        # [B, L, C]
        x = self.norm(x)             # [B, L, C]
        #x = x.transpose(1, 2)        # [B, C, L]
        return x

# 1D ConvNeXtV2 model: ~16.9 million parameters for [64, 128, 256, 512], [3, 3, 18, 3], stem kernel_size=3, Block kernel_size=3
class ConvNeXtV2_1D_ECG(nn.Module):
    def __init__(self, input_channels=12, num_classes=2):
        super().__init__()
        dims = [64, 128, 256, 512]  # Small dimensions [64, 128, 256, 512], [96, 192, 384, 768] try larger dims like [128, 256, 512, 1024] if data and compute allow
        stages = [3, 3, 18, 3]      # Small block counts [2, 2, 6, 2], try larger stages like [3, 3, 27, 3] if data and compute allow
        drop_path_rate = 0.1

        self.stem = Stem1D(input_channels, dims[0], kernel_size=3) # tried 4, 7 or 15 (worse)

        self.blocks = nn.ModuleList()
        dp_rates = [drop_path_rate * (i / (sum(stages) - 1)) for i in range(sum(stages))]
        cur = 0

        for i, num_blocks in enumerate(stages):
            stage = []
            for j in range(num_blocks):
                stage.append(ConvNeXtV2Block1D(dims[i], drop_prob=dp_rates[cur], layer_scale_init_value=1e-6))
                cur += 1
            self.blocks.append(nn.Sequential(*stage))

            if i < len(stages) - 1:
                self.blocks.append(Downsample1D(dims[i], dims[i + 1]))

        #self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        #self.norm = Norm1d(dims[-1])
        self.norm = nn.BatchNorm1d(dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential( # head for classification: output logits for 2 classes
            nn.Linear(dims[-1], 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)

        #x = x.transpose(1,2)
        x = self.norm(x)
        #x = x.transpose(1,2)  # (B, C, L) -> (B, L, C) -> (B, C, L)
        x = self.pool(x).squeeze(-1)  # (B, C, 1) -> (B, C)

        x = self.dropout(x)
        x = self.head(x)
        return x
