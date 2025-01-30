# Splice Site Detection Using Chaos Game Representation and Neural Networks

## Introduction
Splice sites are essential genomic regions that define the boundaries between exons and introns in pre-mRNA. Accurate detection of these sites is crucial for understanding gene expression, genome annotation, and identifying disease-related mutations. However, traditional computational methods struggle with the complexity and variability of splice sites in genetic sequences.

This project presents a novel approach that integrates Chaos Game Representation (CGR) with neural networks for splice site detection. CGR is a visualization technique that converts DNA sequences into geometric patterns, effectively capturing sequence characteristics. These representations are then analyzed using artificial neural networks (ANNs) to classify donor and acceptor splice sites. The combination of CGR and ANNs enhances accuracy and efficiency in splice site prediction.

## Project Specifications
- **Programming Language**: MATLAB
- **Data Source**: NN269 dataset from publicly available genomic databases (Ensembl, GENCODE, UCSC Genome Browser)
- **Feature Extraction Method**: Chaos Game Representation (CGR)
- **Machine Learning Model**: Feedforward Artificial Neural Network (ANN)
- **Dataset Preprocessing**:
  - Balanced dataset containing both positive (true splice sites) and negative (non-splice sites) samples
  - Train-test split to evaluate model performance
- **Training Details**:
  - **Donor Splice Site Model**: 10-node feedforward neural network
  - **Acceptor Splice Site Model**: 20-node feedforward neural network
  - Training optimization with cross-entropy loss and Adam optimizer
- **Performance Metrics**:
  - ROC Curve analysis for donor and acceptor models
  - AUC (Area Under Curve) scores:
    - **Donor AUC**: 0.96395
    - **Acceptor AUC**: 0.92285

This project demonstrates the potential of combining CGR and ANNs to improve splice site detection accuracy. Future enhancements may include scaling the model for large genomic datasets and improving generalization for diverse genetic variations.

