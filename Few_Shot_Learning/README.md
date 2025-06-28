# Exercise 01: Few Shot Learning (FSL)

## Overview
This exercise focuses on implementing and understanding Few Shot Learning techniques using different approaches. The exercise is divided into two main parts:
1. Metric Learning with Siamese Networks
2. Few Shot Learning on the Omniglot dataset

## Part 1: Metric Learning
In this part, we implement and compare two fundamental metric learning approaches:

### 1.1 Triplet Loss Implementation
File: `ex01_Part1_Triplet_Final.ipynb`
- Implementation of Triplet Loss for metric learning
- Training a network to learn embeddings using triplet loss
- Visualization and analysis of learned embeddings
- Performance evaluation and comparison

### 1.2 Contrastive Loss Implementation
File: `ex01_Part1_Contrastive_Final.ipynb`
- Implementation of Contrastive Loss
- Siamese network architecture for pair-wise learning
- Training and evaluation procedures
- Comparative analysis with triplet loss approach

## Part 2: Omniglot Challenge
File: `ex01_Part2_omniglot.ipynb`

The Omniglot dataset is often called the "MNIST of few-shot learning" and consists of 1623 different handwritten characters from 50 different alphabets.

### Key Components:
- Data loading and preprocessing for Omniglot
- Implementation of few-shot learning algorithms
- N-way K-shot classification tasks
- Performance evaluation on novel characters

## Getting Started

### Prerequisites
```python
# Required packages
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Installation
1. Clone this repository
2. Install required dependencies:
```bash
pip install torch torchvision numpy matplotlib pandas
```

### Running the Notebooks
1. Start with Part 1 notebooks to understand metric learning:
   - Run `ex01_Part1_Triplet_Final.ipynb` for triplet loss implementation
   - Run `ex01_Part1_Contrastive_Final.ipynb` for contrastive loss implementation
2. Proceed to Part 2 with `ex01_Part2_omniglot.ipynb` for the Omniglot challenge

## Exercise Structure

## Learning Objectives
- Understanding the fundamentals of metric learning
- Implementing and comparing different loss functions (Triplet and Contrastive)
- Practical experience with Few Shot Learning on a real-world dataset
- Analyzing and visualizing embedding spaces
- Handling limited data scenarios in deep learning

## Tips for Success
1. Start with the theoretical understanding of metric learning concepts
2. Pay attention to the implementation details of loss functions
3. Experiment with different hyperparameters
4. Analyze the visualization of embeddings to understand model behavior
5. Compare the performance of different approaches



