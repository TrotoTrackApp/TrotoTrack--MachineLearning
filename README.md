[//]: # (Image References)

[image0]: ./Images/dataset-sample.png "DatasetSample"
[image1]: ./Images/accuracy.png "Accuracy"
[image2]: ./Images/loss.png "Losses"
[image3]: ./Images/confusion-matrix-test.png "CM Validation"


<p align="center">
  <img src="https://github.com/TrotoTrackApp/TrotoTrack-API/blob/readme/utils/img/logo.png" alt="Nama Alternatif" style="transform: scale(10);"> 
</p>

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Machine Learning Team](#machine-learning-team)

## Introduction <a name="introduction"></a>

<p align="justify">Trototrack is an innovative application designed to streamline and expedite the process of reporting sidewalk defects. It leverages advanced technologies to simplify the identification and monitoring of damaged sidewalks, ensuring a safer and more accessible environment for pedestrians. </p>

## Installation <a name="installation"></a>

To install and set up this project, you can use the following code:

```bash
git clone https://github.com/TrotoTrackApp/TrotoTrack-Machine-Learning.git
pip install -r requirements.txt
```

## Dataset <a name="dataset"></a>
- **Source:** [Dataset Link](https://drive.google.com/drive/folders/1Sd9CN9NAJP24qQrr1IvDX86cCOcHUoxy?usp=sharing) ![DatasetSample][image0]
- **Description:** The model was trained with a dataset of 1,514 sidewalk images collected from Google Maps.
- **Preprocessing:** The images were resized to the corresponding pixels expected by the model and normalized to have pixel values between 0 and 1. This process uses the ImageDataGenerator from the TensorFlow library.

## Model Architecture <a name="model-architecture"></a>

<p align="justify">Our model is built on top of a pre-trained Convolutional Neural Network (CNN), specifically the InceptionV3 architecture, which is pre-trained on the ImageNet dataset. We have used the pre-trained layers for feature extraction and added custom layers for our specific classification task.</p>

- **Pre-trained Base Model:** InceptionV3 (excluding the top layers), initialized with weights from ImageNet.
- **Input Layer:** Accepts 299x299 RGB images.
- **Pre-Trained Layer:** All layers from the InceptionV3 base model (up to the last convolutional block).
- **Conv Layer:** 16 filters of size 3x3, ReLU activation, followed by MaxPooling with pool size 2x2.
- **Flatten Layer:** Flattens the 3D output to 1D.
- **Fully Connected Layer:** 32 neurons, ReLU activation.
- **Dropout Layer:** Dropout rate of 0.1 to prevent overfitting.
- **Output Layer:** Softmax activation with 3 units.

### Hyperparameters

- **Learning Rate:** 0.001
- **Batch Size:** 128
- **Epochs:** 100
- **Optimizer:** Adam

## Training <a name="training"></a>
### Transfer Learning with InceptionV3

<p align="justify">We utilized transfer learning with the InceptionV3 model, leveraging pre-trained weights on the ImageNet dataset. Below are the results from our training process.</p>

### Training and Validation Accuracy

![Accuracy][image1]

<p align="justify">The above plot shows the training and validation accuracy over 100 epochs. The model's training accuracy rapidly increases and stabilizes around 90%, while the validation accuracy follows a similar trend, indicating that the model generalizes well to unseen data.</p>

### Training and Validation Loss

![Losses][image2]

<p align="justify">The plot displays the training and validation loss over 100 epochs. The training loss decreases steadily, and the validation loss follows a similar trend, with some fluctuations. These fluctuations might indicate the model is adjusting to the validation data, but the overall trend suggests that the model is learning effectively without significant overfitting.</p>

### Confusion Matrix

![CM Validation][image3]

<p align="justify">The confusion matrix provides a detailed breakdown of the model's performance on the test set. The diagonal values represent correct classifications, while off-diagonal values indicate misclassifications. From the matrix, we observe that the model performs well on the majority of classes, with the highest misclassification occurring between specific classes.</p>

### Training Procedure

The training procedure involved the following steps:

1. **Pre-trained Model:** We used the InceptionV3 model pre-trained on ImageNet, excluding the top layers.
2. **Custom Layers:** Added a global average pooling layer, a fully connected layer with 16 neurons and ReLU activation, a dropout layer with a rate of 0.1, and a final softmax layer with 3 units.
3. **Frozen Layers:** Initially, all layers of the InceptionV3 model were frozen to leverage the pre-learned features.
4. **Fine-Tuning:** After the initial training, the top 2 inception blocks of InceptionV3 were unfrozen and fine-tuned with a lower learning rate.
5. **Hyperparameters:** We used a learning rate of 0.001. The model was trained with a batch size of 128 over 100 epochs using the Adam optimizer.

By following this procedure, we ensured that the model could effectively learn and adapt to our specific classification task while benefiting from the robust feature extraction capabilities of the pre-trained InceptionV3 model.


## Machine Learning Team <a name="machine-learning-team"></a>

| Name                           | University	                                         | 
| :----------------------------- | :---------------------------------------------------| 
|	Khibran Muhammad Akbar         | Universitas Telkom                                  |
|	Steven Tulus Parulian Elluya Sitompul          | Universitas Udayana                                 |
|	 I Gede Satyananda Gautama          | Universitas Udayana                                 |
