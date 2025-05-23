# Traversable Path Semantic Segmentation

This project implements a semantic segmentation pipeline to identify traversable paths in images. The pipeline includes data preprocessing, model training, evaluation, and testing. It supports multiple segmentation models, including custom CNNs, U-Net with ResNet encoders, SegNet, and SegFormer.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Testing](#testing)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Data Preprocessing**: Organizes raw datasets into a structured format and generates binary ground truth masks for traversable paths.
- **Model Training**: Supports training with early stopping and checkpointing.
- **Evaluation**: Computes metrics such as IoU, pixel accuracy, and F1 score.
- **Testing**: Generates predictions for individual images and visualizes results.
- **Custom Models**: Includes implementations of SimpleSegmentationCNN, UNetResNet, SegNet, and SegFormer.

---

## Project Structure

├── raw_dataset/ # Raw input dataset  
├── processed_dataset/ # Preprocessed dataset  
├── checkpoints/ # Model checkpoints  
├── logs/ # Training logs for TensorBoard  
├── src/ # Source code  
│ ├── dataset.py # Dataset class for loading data  
│ ├── dataloader.py # DataLoader wrapper  
│ ├── model.py # Model definitions  
├── utils/ # Utility scripts  
│ ├── create_folder_structure.py # Organizes raw dataset  
│ ├── create_binary_gt.py # Generates binary ground truth masks  
│ ├── preprocess_data.py # Preprocessing pipeline  
├── evaluation/ # Evaluation scripts  
│ ├── evaluation_methods.py # Metrics computation  
├── train.py # Training script  
├── evaluate.py # Evaluation script  
├── test.py # Testing script  
├── environment.yml # Conda environment configuration  
├── README.md # Project documentation  

---

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
2. conda env create -f environment.yml
3. conda activate seg_env

 note:- Ensure the raw dataset is placed in the raw_dataset/ directory.

4. set preprocess_data = true (first time run only)
4. python3 train.py

---

## Usage

### Data Preprocessing ###
Run the preprocessing pipeline to organize the dataset in the folder structure that will be used by the training pipeline and generate binary ground truth masks. set "preprocess_data = true" for this

### Training ###
Modify the train.py file to select the desired model (e.g., SimpleSegmentationCNN, UNetResNet, SegNet, or SegFormer).
Training logs are saved in the logs/ directory and can be visualized using TensorBoard:
<pre> tensorboard --logdir logs/segmentation </pre>

### Evaluation ###
Evaluate the model on the validation dataset:
<pre> python evaluate.py </pre>

This script computes average IoU and pixel accuracy for the validation set. The evaluation results are printed in the console.

### Testing ###
Test the model on a single image:

<pre> python test.py </pre> 
This script generates a predicted segmentation mask and visualizes the results.

---

## Models
The following models are implemented in src/model.py:

SimpleSegmentationCNN: A lightweight U-Net-inspired CNN.  
UNetResNet: U-Net with a ResNet-34 encoder.  
SegNet: A VGG-based encoder-decoder architecture.  
SegFormer: A transformer-based segmentation model.  

---

## Evaluation Metrics
The following metrics are computed in evaluation/evaluation_methods.py:

1. IoU (Intersection over Union): Measures the overlap between predicted and ground truth masks.
2. Pixel Accuracy: Measures the percentage of correctly classified pixels.
3. F1 Score: Harmonic mean of precision and recall.

---

## Results
The following table summarizes the performance of different models:

| Model                 | Average IoU (%) | Average Pixel Accuracy (%) | Average F1 Score (%) |Model Size (MB)   |    
|-----------------------|-----------------|----------------------------|----------------------|------------------|
| SimpleSegmentationCNN | 78.53           | 92.25                      | 86.11                | ~ 1              |
| UNetResNet            | 89.09           | 96.43                      | 92.47                | ~ 280            |
| SegNet                | 86.62           | 95.57                      | 91.08                | ~ 340            |
| SegFormer             | 90.86           | 97.17                      | 93.48                | ~ 45             |

---

## Video Comparisons

Below are the video comparisons of different models on the `stuttgart_01` scene from the Cityscapes dataset. These videos demonstrate the segmentation performance of each model, highlighting that **SegFormer** achieves the best results in terms of accuracy and visual quality.

| Model                 | Video Link                                                                 |  
|-----------------------|--------------------------------------------------------------------------- |  
| SimpleSegmentationCNN | [Watch Video](https://youtu.be/t3i9m9SugbI)                                |  
| UNetResNet            | [Watch Video](https://youtu.be/SBOQ8OWKl34)                                |  
| SegNet                | [Watch Video](https://youtu.be/ITxU31A5Dgc)                                |  
| SegFormer             | [Watch Video](https://youtu.be/QYtwCOkoq8U)    

---

## Conclusion
This project successfully implements a semantic segmentation pipeline for identifying traversable paths in images. The pipeline supports multiple models, including both traditional CNN-based architectures and transformer-based models like SegFormer. The results demonstrate that SegFormer achieves the best performance in terms of IoU, pixel accuracy, and F1 score, while maintaining a relatively small model size.

The modular design of the codebase allows for easy experimentation with different models, datasets, and evaluation metrics. This project can be extended further by incorporating additional datasets, optimizing model architectures, or deploying the trained models for real-world applications such as autonomous navigation or robotics.

Feel free to contribute to this project or use it as a foundation for your own semantic segmentation tasks!

---

## Acknowledgments
1. The project uses the Cityscapes dataset for semantic segmentation.
2. The SegFormer model is based on the Hugging Face Transformers library.
3. Pretrained weights for ResNet and VGG are sourced from TorchVision.

