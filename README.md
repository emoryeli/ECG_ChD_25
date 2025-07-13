# Team ECG_ChD_25
George B. Moody PhysioNet Challenge 2025

For training, our code uses data from all three sources: Sami-Trop, PTB-XL and 'CODE-15%'. 

Our algorithm is based on a one-dimensional (1D) EfficientNet B3 and ConvNeXt V2 deep neural networks (DNN). We trained the 1D DNN model with labeled data: 1000 from Sami-Trop, 14000 from CODE-15% and 10920 for PTB-XL. Class weights [1, 19.0] are applied to negative and positve cases to address class imbalance. Label smoothing (0.1) is applied only to the weakly labeled CODE-15% data. A custom loss function combining a Generalized Cross-entropy loss (or Focal loss) and a Softmax approximated differentiable TPR@5% loss, AdamW optimizer over 10 epochs, initial learning rate of 2e-4, dropout rate of 0.3, and weight_decay of 1e-2 were used. Dynamic learning rate reduction on plateau was used to enable more effective training convergence. A 5-fold cross validation was employed and the final model was chosen as the best performing model from each fold.

In the future, we will develop two additional DNN models: InceptionTime (multi-scale convolutional neural network) and TimesNet (attention-based) adapted for binary time series classification. We will compare the three models' performances. 

We will also explore the use of gradient-weighted class activation mapping (Grad-CAM++) and XRAI to locate potentially human-interpretable 12-lead ECG features thus far unidentified for chronic Chagas cardiomyopathy.

If time permits, we will also explore converting 12 lead ECG data to 2D ECG images and fine-tune some pretrained Pytorch Vision models such as ViT or RegNet. 

