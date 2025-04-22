# Landmark-Detection
Landmark classification using modified VGG19 architecture. Processes image dataset, implements batch training with RMSprop optimizer, adds dropout and batch normalization layers, and visualizes top-5 predictions. Uses 80/20 train/validation split with 224×224 pixel images.

This project implements a landmark image classification system using transfer learning with a VGG19 architecture. The system is designed to classify images into different landmark categories based on a pre-existing dataset of landmark images.
Project Overview
The code performs the following key functions:

Data Loading and Exploration:

Loads landmark image data from a CSV file containing image IDs and landmark IDs
Filters the dataset to include only specific image IDs (those starting with '00')
Analyzes the distribution of images across different landmark classes
Visualizes the distribution of samples per landmark category

Image Processing:

Creates functions to load images from a directory structure
Implements image preprocessing, including resizing to 224×224 pixels and normalization
Displays sample images from random classes for visual inspection

Model Architecture:

Uses a modified VGG19 architecture (without pre-trained weights)
Adds BatchNormalization for improved training stability
Implements dropout layers (0.5) to prevent overfitting
Adds a final dense layer with softmax activation for multi-class classification

Training Pipeline:

Splits data into training (80%) and validation (20%) sets
Implements a batch processing system for memory-efficient training
Uses RMSprop optimizer with custom learning rate
Trains the model for one epoch with the sparse categorical cross-entropy loss function

Evaluation and Visualization:

Evaluate model performance on the validation set
Tracks correct and incorrect predictions
Visualizes the top 5 most confident correct predictions
Displays images alongside their predicted and actual classes

This project demonstrates the application of convolutional neural networks for landmark recognition, which has applications in tourism, augmented reality, and geographic information systems.
