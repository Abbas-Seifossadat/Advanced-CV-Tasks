# Advanced Computer Vision Tasks

This repository contains implementations of advanced computer vision tasks completed as part of the course materials from [@advanced_cv_1403](https://github.com/MhmudAlpurd/advanced_cv_1403).

## Overview

This repository focuses on implementing cutting-edge computer vision techniques that go beyond traditional image classification and object detection. The tasks demonstrate advanced approaches in machine learning and computer vision that handle scenarios with limited or no training data.

## Tasks Implemented

### 1. Few Shot Learning
Few-shot learning addresses the challenge of learning from a limited number of training examples. This implementation demonstrates:
- Model architecture for few-shot learning
- Training with limited data samples
- Evaluation on novel classes with few examples
- Techniques for feature extraction and metric learning

### 2. Zero Shot Learning
Zero-shot learning enables recognition of unseen classes without any training examples. Key aspects covered:
- Semantic embedding space creation
- Visual-semantic mapping
- Inference on unseen classes
- Implementation of ZSL architectures

### 3. Fine Grained Image Classification
Fine-grained image classification focuses on distinguishing between subtle variations within a category. Implementation includes:
- Detailed feature extraction techniques
- Attention mechanisms for focusing on discriminative parts
- Handling of subtle inter-class variations
- Performance optimization strategies

### 4. Satellite Image Analysis: Spectral Indices (NDWI, MNDWI, NDVI) with Google Earth Engine
This project focuses on the application of spectral indices for environmental analysis using satellite imagery. It demonstrates the calculation and visualization of:

- **Normalized Difference Water Index (NDWI):** For delineating open water features.

- **Modified Normalized Difference Water Index (MNDWI):** For enhanced water feature extraction, particularly in urban areas, by suppressing built-up land noise.

- **Normalized Difference Vegetation Index (NDVI):** For quantifying vegetation greenness and health.

The implementation utilizes Google Earth Engine (GEE) and Landsat 8 satellite imagery to perform these calculations and generate visual outputs.


### 5. YOLOv12 Object Detection Notebooks
This section explores advanced object detection using YOLOv12. It includes two notebooks focused on inference and training:

- **Notebook 1 – Inference and Visualization:**  
  Evaluates YOLOv12-small, medium, and large models on a sample image. It compares runtime performance and visualizes results using bounding boxes and labels, including filtering for specific classes (e.g., person). This serves as a performance benchmark and inference demonstration.

- **Notebook 2 – Training on Car Components Dataset:**  
  Provides a full pipeline for fine-tuning YOLOv12 on a Roboflow car components dataset. It covers:
  - Dataset inspection and annotation visualization
  - Albumentations-based data augmentation
  - Training and evaluation of `yolov12l.pt`
  - Real-world deployment on images of Iranian cars


## Contributing

Contributions to improve the implementations are welcome. Please feel free to submit pull requests or create issues for any bugs or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Course materials from [@advanced_cv_1403](https://github.com/MhmudAlpurd/advanced_cv_1403)


## Contact

For any questions or discussions about the implementations, please open an issue in the repository.