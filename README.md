# Classifying_X-Ray_Images_using_PyTorch

## Background:
Pneumonia is one of the leading respiratory illnesses worldwide, and its timely and accurate diagnosis is essential for effective treatment. Manually reviewing chest X-rays is a critical step in this process, and AI can provide valuable support by helping to expedite the assessment.

## Goal:
As a consultant data scientist, I will test the ability of a deep learning model to distinguish pneumonia cases from normal images of lungs in chest X-rays. By fine-tuning a pre-trained convolutional neural network, specifically the ResNet-18 model, the task is to classify X-ray images into two distinct categories: normal lungs and those affected by pneumonia.

## Why a pre-trained model:
By leveraging an already trained model (pre-trained on ImageNet), we can utilize its pre-computed feature extractors, allowing us to build an accurate classifier faster and with significantly fewer computational resources than training from scratch.

## Methodology:
### 1) Load and adjust the ResNet-18 model
1.1) Loading the architecture and the pre-trained parameters of a ResNet-18 model. 
1.2) Freeze their gradients 
1.3) Change the final layer for binary classification.

### 2) Write the training loop
Fine-tuning the ResNet-18 model using appropriate loss function and the provided training data

### 3) Train and evaluate your model
3.1) Run the training loop for 3 epochs updating the parameters of the network meant for fine-tuning. 
3.2) Calculate the accuracy and F1-score on the test set.

## Results
Epoch [1/3], train loss: 1.3915, train acc: 0.4567
Epoch [2/3], train loss: 0.8973, train acc: 0.4633
Epoch [3/3], train loss: 0.9199, train acc: 0.5033

Test accuracy: 0.825
Test F1-score: 0.831

## Key Technical Details

### Transfer Learning Backbone: 
ResNet-18 pre-trained on ImageNet weights, providing robust low-level edge and texture detectors suited for medical imaging texture analysis.

### Loss Function: 
BCEWithLogitsLoss, which combines a sigmoid activation layer and binary cross-entropy in a single class for numerical stability.

### Optimizer: 
Adam Optimizer configured with a learning rate of $0.01$ targeting the parameters of the final custom classification layer (model.fc.parameters()).

### Image Normalization: 
Input images are normalized using ImageNet dataset statistics.

## Future Improvements
- Unfreezing Layers (Fine-Tuning Deeper): Unfreeze layers closer to the convolutional backbone to let the network adapt deeper domain-specific features to chest X-rays.
- Data Augmentation: Introduce rotation, zooming, and horizontal flipping via torchvision.transforms to improve model generalization and prevent overfitting.
- Hyperparameter Tuning: Experiment with learning rate schedulers, weight decay, and a higher number of training epochs.
