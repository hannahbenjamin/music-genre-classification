# Music Genre Classification Project

Welcome to the Music Genre Classification Project! This repository contains the code and resources for a deep learning-based approach to classifying music genres. The project leverages the power of deep learning, convolutional neural networks (CNNs), and recurrent neural networks (RNNs) to automatically classify music genres from audio data, enhancing music discovery and analysis.

## Project Overview
Music has played a crucial role in human culture and personal expression throughout history. Given its significance, this project aims to improve the experience of navigating vast digital music libraries and enhance music discovery through automated genre classification. The core objective is to develop a system that can accurately identify the genre of a given piece of music by analyzing its audio features.

## Approach
### 1. Data Collection and Preparation:
Dataset: The GTZAN dataset, available on Kaggle, is used. It includes 10 genres: blues, classical, reggae, country, disco, hiphop, jazz, metal, pop, and rock. Each genre consists of 100 audio clips of 30 seconds each, in WAV format.
Libraries: librosa is used for primary audio loading, and pydub handles any loading exceptions, ensuring robustness against format inconsistencies.

### 2. Model Architecture:
CNNs: Convolutional Neural Networks are employed to learn patterns from visual representations of music, such as spectrograms and tempograms. These networks process the data through convolution, ReLU, dropout, and pooling layers to abstract and reduce spatial variability.
RNNs: Recurrent Neural Networks are used to capture temporal features in audio signals, focusing on musical dynamics and rhythmic characteristics over time.
<img width="706" alt="Model Architecture" src="https://github.com/user-attachments/assets/979a9185-c1f7-4633-802d-878cf6bce9fa">

### 3. Training and Evaluation:
The model was trained, validated, and tested using the dataset. The training involved tuning various hyperparameters to achieve the best performance.
Performance: The final model achieved accuracies of 77.80%, 61.81%, and 79.60% in training, validation, and testing datasets, respectively.
<img width="781" alt="Train vs Validation Accuracy" src="https://github.com/user-attachments/assets/53a37677-800d-42c6-9fe7-cf7ce8551801">

