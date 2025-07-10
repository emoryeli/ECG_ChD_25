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
from math import gcd
import scipy.signal as sgn
import pywt # wavelet filter for baseline removal
import matplotlib.pyplot as plt

from helper_code import *

# Select device (GPU or MPS or CPU): 16 vCPUs 64GB RAM on AWS
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()  # helpful after large model or dataset loads
elif device.type == 'cpu':
    torch.set_num_threads(os.cpu_count())  # or 16

THRESHOLD_PROBABILITY = 0.8 # above this threshold, the model predicts Chagas disease
source_string = '# Source:' # used to get the source of the data in the traiing set: CODE-15%, PTB-XL, or SaMi-Trop
SAMI = 'SaMi-Trop' # used to limit SaMi data size in the training set
PTBXL = 'PTB-XL' # used to limit PTB-XL data size in  the training set
CODE15 = 'CODE-15%' # used to limit CODE-15% data size in the traiing set 
ECG_len = 4096 #  4096 or 5000 or 2934: ?? double-check: all Chagas positives in CODE-15% are 2934 long
EPOCHS = 10 # number of epochs to train the model

class ECGDataset(Dataset):
    def __init__(self, records, labels, smoothing_flags):
        self.records = records
        self.labels = labels
        self.smoothing_flags = smoothing_flags

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal = extract_ECG(self.records[idx])
        #assert not torch.isnan(signal).any(), "NaN in input signal!"
        #assert not torch.isinf(signal).any(), "Inf in input signal!"    
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        smooth = torch.tensor(self.smoothing_flags[idx], dtype=torch.bool)
        return signal, label, smooth

# custom Focal loss function with weighted label and label smoothing on CODE-15% data 
def custom_focal_loss(outputs, targets, smoothing_flags, weight, smoothing=0.1, gamma=1.0):
    # outputs: logits (batch, num_classes)
    # targets: class indices (batch,)
    # smoothing_flags: (batch,)
    # weight: (num_classes,) class weights, e.g., torch.tensor([1.0, 19.0])
    # gamma: focusing parameter
    # smoothing: label smoothing value

    #Clamp outputs to avoid NaN/Inf
    outputs = torch.clamp(outputs, min=-20, max=20)

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
    focal_term = (1.0 - probs) ** gamma

    # Class weights (alpha)
    class_weights = weight.to(device).unsqueeze(0)  # (1, num_classes)

    # Combine all terms
    loss = -(true_dist * focal_term * log_probs * class_weights).sum(dim=1).mean()

    return loss

# Softmax (sigmoid for binary) approximated top-k true positive rate loss that is differentiable
def soft_topk_tpr_loss(outputs, targets, k_frac=0.05, temperature=0.05):
    """
    outputs: (batch,) - logits or scores, higher = more positive
    targets: (batch,) - binary ground truth (0 or 1)
    k_frac: fraction of batch to be considered "top k" (e.g., 0.05 for top 5%)
    temperature: controls sharpness, lower = harder selection
    """

    #Clamp outputs to avoid NaN/Inf
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
    tps = (soft_topk_weights * targets).sum()
    total_positives = targets.sum().clamp(min=1)

    tpr_at_k = tps / total_positives
    loss = 1 - tpr_at_k
    return loss

def joint_loss(
    outputs, targets, 
    smoothing_flags, # bool tensor for label smoothing, shape (batch,), True only for CODE-15% data 
    weight, # tensor([w_neg, w_pos]) for class weights
    smoothing=0.1, # label smoothing for weakly labeled CODE-15% data
    gamma=1.0, # focusing parameter for focal loss
    k_frac=0.05,   # fraction of batch to be considered "top k" (e.g., 0.05 for top 5%)
    temperature=0.05, # temperature for soft_topk_tpr_loss
    alpha=0.3  # tradeoff parameter, 0=only custom_focal_loss, 1=only soft_topk_tpr
):
    """
    alpha: tradeoff parameter, 0=only custom_focal_loss, 1=only soft_topk_tpr
    """

    # 1. Main (classification) loss
    loss_ce = custom_focal_loss(outputs, targets, smoothing_flags, weight, smoothing=smoothing, gamma=gamma)
    # 2. Ranking loss on positive class only (class 1)
    pos_outputs = outputs[:, 1]          # logits for class 1
    pos_labels = (targets == 1).float()  # binary 0/1
    loss_tpr = soft_topk_tpr_loss(
        outputs=pos_outputs,
        targets=pos_labels,
        k_frac=k_frac,
        temperature=temperature
    )
    # 3. Weighted sum
    total_loss = (1 - alpha) * loss_ce + alpha * loss_tpr
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
    MAX_SAMI = 1000 # all positives
    MAX_CODE15 = 15000 # ~1.91% positives but weakly labeled (patient self-reported, not confirmed by a serological test)
    MAX_PTBXL = 9920 # all negatives from Europe not from the endemic region South America
    MAX_TOTAL = 25920 # mutiples of batch_size to avoid last batch size mismatch; this should give Chagas prevalence <= 5% in the training set, hopefully this can finish training in 72 hours on 16vCPUs on aws

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
    data_paths = [os.path.join(data_folder, r) for r in records]
    labels = np.array(labels)
    label_smoothing_flags = np.array(label_smoothing_flags)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_val_loss_overall = float('inf')
    best_val_challenge_score_overall = 0.0
    best_model_state_overall = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels), 1):
        if verbose:
            print(f"\nStarting Fold {fold}...")

        train_records = [data_paths[i] for i in train_idx]
        val_records = [data_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_smoothing_flags = label_smoothing_flags[train_idx]
        val_smoothing_flags = label_smoothing_flags[val_idx]  # not actually needed for validation

        train_dataset = ECGDataset(train_records, train_labels, train_smoothing_flags)
        val_dataset = ECGDataset(val_records, val_labels, val_smoothing_flags)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, drop_last=True)

        model = EfficientNetB3_1D().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        weight = torch.tensor([1.0, 19.0])  # assuming 5% positive -> 1:19 imbalance

        best_val_loss = float('inf')
        best_val_challenge_score = 0.0

        # training loop
        for epoch in range(EPOCHS): 
            model.train()
            total_loss = 0
            for X, y, smooth_flag in train_loader:

                optimizer.zero_grad()
                outputs = model(X.to(device)).contiguous()  # logits from model, shape (batch, 2)

                outputs = torch.clamp(outputs, min=-20, max=20)

                loss = joint_loss(
                    outputs,         # logits from model, shape (batch, 2)
                    y.to(device),         # int64 tensor, shape (batch,)
                    smooth_flag.to(device), # bool tensor, shape (batch,)
                    weight,          # tensor([w_neg, w_pos])
                    smoothing=0.1,  # label smoothing only for weakly labeled CODE-15% data, if too many FP's, decrease smoothing
                    gamma=0.5, # focusing parameter for focal loss
                    k_frac=0.05, # k for soft_topk_tpr_loss
                    temperature=0.05, # if too many FP's, increase temperature
                    alpha=0.3, # if too many FP's, decrease alpha
                ) 

                loss.backward()

                # Gradient clipping
                total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

                total_loss += loss.item() * X.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0
            val_outputs = [] #predicted probabilities, float64
            val_targets = [] #labels, 0 or 1
            
            # validation loop
            with torch.no_grad():
                for X, y, smooth_flag in val_loader:
                    outputs = model(X.to(device))
                    outputs = torch.clamp(outputs, min=-20, max=20)
                    # TPR@5% loss for validation:
                    loss = joint_loss(
                        outputs,         # logits from model, shape (batch, 2)
                        y.to(device),         # int64 tensor, shape (batch,)
                        smooth_flag.to(device), # bool tensor, shape (batch,)
                        weight,          # tensor([w_neg, w_pos])
                        smoothing=0.1,  # no label smoothing for validation
                        gamma=0.5, # focusing parameter for focal loss, not used for validation
                        k_frac=0.05, # k for soft_topk_tpr_loss
                        temperature=0.05, # temperature for soft_topk_tpr_loss
                        alpha=0.3 # only use TPR@5% performance for validation, not cross entropy loss
                    )
                    val_loss += loss.item() * X.size(0)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # probs for class 1: Chagas
                    val_outputs.extend(probs.tolist())
                    val_targets.extend(y.cpu().numpy().tolist())
            avg_val_loss = val_loss / len(val_loader.dataset)

            scheduler.step(avg_val_loss)

            #f1 = compute_f_measure(val_labels, (np.array(val_outputs) > THRESHOLD_PROBABILITY).astype(int))
            #best_threshold, best_f1 = compute_f1_and_thres(val_labels, np.array(val_outputs))
            challenge_score = compute_challenge_score(np.array(val_labels), np.array(val_outputs))
            #auroc, auprc = compute_auc(np.array(val_labels), np.array(val_outputs))

            gc.collect()  # free memory after each epoch

            if challenge_score > best_val_challenge_score: # use challenge score instead of validation loss
                best_val_challenge_score = challenge_score
                best_model_state = model.state_dict()

            os.makedirs(model_folder, exist_ok=True)
            torch.save({'model_state_dict': best_model_state}, os.path.join(model_folder, 'model.pt'))

        if best_val_challenge_score > best_val_challenge_score_overall: # use challenge score instead of validation loss
            best_val_challenge_score_overall = best_val_challenge_score
            best_model_state_overall = best_model_state

    torch.save({'model_state_dict': best_model_state_overall}, os.path.join(model_folder, 'model.pt'))

def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_folder, 'model.pt'))

def load_model(model_folder, verbose):
    model = EfficientNetB3_1D()
    checkpoint = torch.load(os.path.join(model_folder, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_model(record, model, verbose):
    x = extract_ECG(record).unsqueeze(0)
    output = model(x)
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
        g = gcd(int(target_fs), int(fs))
        up = target_fs // g
        down = fs // g
        # Resample each lead separately using resample_poly
        signal = np.array([resample_poly(lead, up, down) for lead in signal])

    # Truncate or pad to fixed length (e.g. max_len = 2934 samples for 7.2s at 400 Hz)
    # ECG_len = 4096 or 5000 or 2934, some SaMi-Trop and CODE-15% data have 2934 samples, all Chagas positves in CODE-15% are 2934 long
    if signal.shape[1] < ECG_len:
        pad_width = ECG_len - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)))
    else: # random cropping for data augmentation
        signal = random_crop(signal, target_length=ECG_len)

    signal = ECG_preprocess(signal, sample_rate=target_fs, powerline_freqs=[60, 50], bandwidth=[0.05, 150], augment=True, target_length=ECG_len)
    assert not np.isnan(signal).any(), "NaN found after ECG_preprocess"

    return torch.tensor(signal.copy(), dtype=torch.float32)

# ECG pre-processing functions:
# some adapted from https://github.com/antonior92/ecg-preprocessing/
def remove_baseline_wavelet_filter(ecg_12lead, wavelet='sym8', level=7): # or 8
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

def bandpass_filter(bandwidth=[0.05, 150], sample_rate=400):
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

def ECG_preprocess(signal, sample_rate=400, powerline_freqs=[60, 50], bandwidth=[0.05, 150], augment=False, target_length=ECG_len):
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

    # Remove edge artifacts: the first and last 0.25 seconds 
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

# Squeeze-and-Excitation block for 1D data
class SEBlock1D(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv1d(in_channels, reduced_channels, 1)
        self.fc2 = nn.Conv1d(reduced_channels, in_channels, 1)
    def forward(self, x):
        s = x.mean(-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

# MBConv1D block
class MBConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch) and (stride == 1)
        self.expand = (expand_ratio != 1)
        if self.expand:
            self.expand_conv = nn.Conv1d(in_ch, mid_ch, 1, bias=False)
            self.bn0 = nn.BatchNorm1d(mid_ch)
        else:
            mid_ch = in_ch
        self.dwconv = nn.Conv1d(mid_ch, mid_ch, kernel_size, stride, kernel_size//2, groups=mid_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.se = SEBlock1D(mid_ch, se_ratio=se_ratio) if se_ratio else nn.Identity()
        self.pwconv = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        identity = x
        if self.expand:
            x = self.act(self.bn0(self.expand_conv(x)))
        x = self.act(self.bn1(self.dwconv(x)))
        x = self.se(x)
        x = self.bn2(self.pwconv(x))
        if self.use_residual:
            x = x + identity
        return x

# EfficientNet-B3 (original) architecture adapted for 1D 12-lead ECG data: ~43.7 million trainable parameters
class EfficientNetB3_1D(nn.Module): 
    def __init__(self, input_channels=12, num_classes=2):
        super().__init__()
        # Official B3 configuration (from paper/code)
        chs    = [40, 72, 120, 208, 352, 1536]
        reps   = [3,   5,   3,   5,   8,   1]
        ksizes = [3,   3,   5,   3,   5,   3]
        strides= [1,   2,   2,   2,   1,   2]
        expand = [1,   6,   6,   6,   6,   6]
        se_ratios = [0.25] * 6

        # Stem (input_channels -> chs[0], kernel_size=3, stride=2)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, chs[0], kernel_size=3, stride=2, padding=2, bias=False), # try kernel_size=3, 5, 15, 17: 
            nn.BatchNorm1d(chs[0]),
            nn.SiLU(inplace=True)
        )

        blocks = []
        in_ch = chs[0]
        for idx in range(len(chs)):
            for i in range(reps[idx]):
                s = strides[idx] if i == 0 else 1
                blocks.append(
                    MBConv1D(
                        in_ch, chs[idx],
                        kernel_size=ksizes[idx],
                        stride=s,
                        expand_ratio=expand[idx],
                        se_ratio=se_ratios[idx]
                    )
                )
                in_ch = chs[idx]
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(in_ch, chs[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(chs[-1]),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(chs[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
