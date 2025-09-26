# Team ECG_ChD_25
George B. Moody PhysioNet Challenge 2025
https://moody-challenge.physionet.org/2025/

For training, our code uses ECG data from three public sources: Sami-Trop, PTB-XL and 'CODE-15%'. 

Our algorithm is based on a one-dimensional (1D) ConvNeXt deep neural networks (DNN) [1]. We trained the 1D ConvNeXt model with labeled 12 lead ECG data: 1631 from Sami-Trop, 21000 from PTB-XL and 28569 Chagas negatives from CODE-15%. Class weights [1, 9.0] are applied to negative and positve cases to address class imbalance. Label smoothing (0.02) is applied only to the weakly labeled CODE-15% data. We also oversampled minority class to 50% per epoch. A custom loss function combining Focal loss with a pair-wise ranking (hinge) loss, AdamW optimizer over 12 epochs, initial learning rate of 2e-4, dropout rate of 0.3, and weight_decay of 1e-3 were used. We maintained an exponential moving average (EMA) of the weights (decay = 0.999, BN buffer decay = 0.9) during training. Cosine annealing LR scheduler (T_max = 12 epochs, minimum LR = 5e-6) was used to enable more effective training convergence. A 5-fold cross validation was employed and the final model was chosen as a logit-averaged ensemble of the best-performing checkpoint from each fold.

The model has ~16.9M learnable parameters. We implemented it in Python 3.10.1 using PyTorch 2.3.0 and scikit-learn 1.6.0. The training was done on a MacBook Pro M4 Max (14-core CPU, 32-core GPU, 36GB RAM, PyTorch MPS backend). We used FP32 (full precision) training without mixed precision.

In the future, we will develop two additional DNN models: InceptionTime (multi-scale convolutional neural network) and TimesNet (attention-based) adapted for binary time series classification. We will compare the three models' performances. 

We will also explore the use of gradient-weighted class activation mapping (Grad-CAM++) and XRAI to locate potentially human-interpretable 12-lead ECG features thus far unidentified for chronic Chagas cardiomyopathy.

If time permits, we will also explore converting 12 lead ECG data to 2D ECG images and fine-tune some pretrained large Vision models such as ViT, Swin Transformer or RegNet. 

[1] Z. Liu, et al., “A ConvNet for the 2020s,” Mar. 02, 2022, arXiv: arXiv:2201.03545. doi: 10.48550/arXiv.2201.03545
