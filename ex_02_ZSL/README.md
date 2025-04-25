# Exercise 02: Zero Shot Learning (ZSL)

## Overview
This exercise focuses on implementing a generative-based model for Zero Shot Learning (ZSL). ZSL is a challenging task in computer vision where we aim to recognize objects from classes that were never seen during training by leveraging semantic information.


## Zero Shot Learning Implementation
The main implementation (`ex02_ZSL.ipynb`) focuses on:

### 1. Model Architecture
- Generative-based approach for ZSL
- Feature extraction backbone
- Semantic embedding module
- Generative components for unseen classes

### 2. Key Components
- **Feature Extractor**: Extracts visual features from images
- **Semantic Encoder**: Processes class descriptions/attributes
- **Generator**: Creates synthetic features for unseen classes
- **Classifier**: Final classification layer

### 3. Training Process
1. Train feature extractor on seen classes
2. Learn semantic-visual mapping
3. Generate synthetic features for unseen classes
4. Train final classifier

### 4. Evaluation
- Performance metrics on unseen classes
- Analysis of generated features
- Comparison with baseline methods

## Supporting Files

### Loss Functions (`ex02_Loss.ipynb`)
A supplementary notebook containing implementations of:
- Contrastive Loss: For learning discriminative feature spaces
- Cosine Similarity Loss: For semantic-visual alignment

## Requirements
```python
numpy
torch
torchvision
matplotlib
scikit-learn
```

## Usage
1. Start with the main ZSL implementation:
```bash
jupyter notebook ex02_ZSL.ipynb
```

2. Review loss function implementations:
```bash
jupyter notebook ex02_Loss.ipynb
```