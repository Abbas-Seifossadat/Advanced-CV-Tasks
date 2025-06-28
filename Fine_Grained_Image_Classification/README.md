# Exercise 03: Fine-Grained Image Classification (FGIC)

## Overview
This exercise focuses on implementing Fine-Grained Image Classification using the CUB (Caltech-UCSD Birds) dataset. FGIC is a challenging computer vision task that requires distinguishing between visually similar subcategories within a broader category, such as different species of birds.


## CUB-200-2011 Dataset
The Caltech-UCSD Birds-200-2011 dataset contains:
- 200 bird species
- 11,788 images
- Annotated with:
  - Bounding boxes
  - Part locations
  - Attribute labels

## 🔑 Key Components

### 1. Data Preprocessing
- Image loading and normalization
- Bounding box handling
- Part annotation processing
- Data augmentation strategies

### 2. Model Architecture
Our implementation includes:
- Backbone CNN for feature extraction
- Attention mechanism for focusing on discriminative parts
- Part-based learning modules
- Fine-grained classification head

### 3. Training Pipeline
- Multi-stage training process
- Part localization
- Feature learning
- Classification optimization

## 📊 Implementation Details

### Model Features
- **Attention Mechanism**: Learn to focus on discriminative bird parts
- **Part-based Learning**: Utilize bird part annotations
- **Multi-scale Processing**: Handle varying scales of visual features
- **Fine-grained Feature Extraction**: Capture subtle differences between species

### Training Strategy
1. Backbone network pre-training
2. Part detector fine-tuning
3. Attention mechanism training
4. End-to-end fine-tuning

## 🛠️ Requirements
```python
torch>=1.7.0
torchvision>=0.8.0
numpy
pandas
matplotlib
PIL
scikit-learn
```

## 🚀 Getting Started

1. **Setup Environment**
```bash
pip install -r requirements.txt
```

2. **Download Dataset**
```bash
# Instructions for downloading CUB-200-2011 dataset
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
```

3. **Run the Notebook**
```bash
jupyter notebook "FGIC on CUB Dataset.ipynb"
```

## 📈 Expected Outcomes

### Performance Metrics
- Top-1 Accuracy
- Top-5 Accuracy
- Part Localization Accuracy
- Confusion Matrix Analysis

### Visualization
- Attention Maps
- Part Localization Results
- Feature Embeddings
- Misclassification Analysis

## Tips for Success
1. Pay attention to data preprocessing and augmentation
2. Carefully tune the attention mechanism
3. Monitor part localization accuracy
4. Use appropriate learning rate scheduling
5. Implement proper validation strategies

## Evaluation Criteria
- Classification accuracy on test set
- Part localization performance
- Attention mechanism effectiveness
- Model efficiency and runtime
- Quality of visualizations

