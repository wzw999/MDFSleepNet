# MDFSleepNet

code for "Sleep Stage Specificity to Window Length Variations: A Decision Fusion Strategy for Enhanced Scoring"

## Overview

MDFSleepNet is a deep learning framework for automatic sleep stage classification that combines multi-scale temporal features from different time segments (1s and 30s) to improve classification accuracy.

## Features

- Multi-scale temporal feature extraction
- ResNet-based CNN architecture
- Cross-validation evaluation
- Support for multiple datasets (Sleep-EDF-2013, Sleep-EDF-2018, ISRUC)
- Comprehensive evaluation metrics and confusion matrices

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset Structure

The project expects preprocessed data in NPZ format with the following structure:
- `Fold_len`: Number of samples in each fold
- `Fold_data`: EEG data for each fold
- `Fold_label`: Sleep stage labels for each fold

## Usage

1. Prepare your dataset using the preprocessing scripts in `utils/`
2. Modify the dataset paths in the main scripts
3. Run the training and evaluation:
   ```bash
   python MFE_seg.py  # For single model training
   python MFE_seg_c.py  # For fusion model evaluation
   ```

## Sleep Stages

The model classifies 5 sleep stages:
- W: Wake
- N1: NREM Stage 1
- N2: NREM Stage 2
- N3: NREM Stage 3
- REM: Rapid Eye Movement Sleep

## File Structure

- `MFE_seg.py`: Main training script for individual models
- `MFE_seg_c.py`: Fusion model evaluation script
- `Utils.py`: Utility functions for evaluation and visualization
- `DataGenerator.py`: Data loading and k-fold cross-validation
- `utils/`: Dataset preprocessing scripts
