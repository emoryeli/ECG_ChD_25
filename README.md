# Team ECG_ChD_25
George B. Moody PhysioNet Challenge 2025

For training, our code automatically removes data from the source 'CODE-15%' if they are present in the training dataset, because the current model doesn't work well with the weakly-labeled CODE-15% data.

Our algorithm is based on a one-dimensional (1D) ConvNeXt V2 deep neural network (DNN) enhanced by a Transformer Encoder layer for global attention. We trained the 1D DNN model with labeled SaMi-Trop and PTB-XL 12-lead ECG datasets. Cross-entropy loss, AdamW optimizer over 20 epochs, initial learning rate of 2e-4, dropout rate of 0.3, and weight_decay of 1e-3 were used. Dynamic learning rate reduction on plateau was used to enable more effective training convergence. A 5-fold cross validation was employed and the final model was chosen as the best performing model from each fold.

In the future, we will develop two additional DNN models: InceptionTime (multi-scale convolutional neural network) and TimesNet (attention-based) adapted for binary time series classification. We will compare the three models' performances.

We will also explore the use of gradient-weighted class activation mapping (Grad-CAM++) and XRAI to locate potentially human-interpretable 12-lead ECG features thus far unidentified for chronic Chagas cardiomyopathy.

We will also try different data pre-processing and augementation techniques to better utilize the CODE-15% dataset.
