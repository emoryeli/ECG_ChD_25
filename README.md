# ECG_ChD_25
PhysioNet 2025 Challenge

For training, our code automatically removes data from source 'CODE-15%' if they are present in the training dataset, because the current model doesn't work well with the weakly-labeled CODE-15% data.

Our algorithm is based on a one-dimensional (1D) ResNet deep neural network (DNN). We trained the 1D ResNet model with labeled SaMi-Trop and PTB-XL 12-lead ECG datasets. Cross-entropy loss, AdamW optimizer over 20 epoch, initial learning rate of 2e-4, dropout rate of 0.5, and weight_decay of 1e-3 were used. Dynamic learning rate reduction on plateau was used to enable more effective training convergence. A 5-fold cross validation was employed and the final model was chosen as the ensemble of the best performing models from each fold.

In the official phase, we will develop two additional DNN models: InceptionTime (multi-scale convolutional neural network) and TimesNet (attention-based) adapted for binary time series classification. We will compare the three models' performances.

We will also explore the use of gradient-weighted class activation mapping (Grad-CAM++) to locate potentially human-interpretable 12-lead ECG features thus far unidentified for chronic Chagas cardiomyopathy.

We will also try different data pre-processing and augementation techniques to better utilize CODE-15% dataset.
