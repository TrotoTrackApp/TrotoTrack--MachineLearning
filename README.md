[//]: # (Image References)

[image0]: ./Images/dataset-sample.png "DatasetSample"
[image1]: ./Images/accuracy.png "Accuracy"
[image2]: ./Images/loss.png "Losses"
[image3]: ./Images/confusion-matrix-test.png "CM Validation"


## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)

## Project Overview <a name="project-overview"></a>

<p align="justify"> The energy research institute “Enerdata” stated that Indonesia's energy sector emissions increased from 470 metric tons of carbon dioxide (MtCO2) in 2015 to 581 MtC02 in 2019. 80% of which is contributed by vehicle emissions as the number of vehicle users increases every year. </p>
<p align="justify">People's reluctance to walk causes the increasing number of vehicle users in Indonesia. Based on Stanford University research, Indonesia has the lowest average steps per day at 3.513. One reason is due to the fact that sidewalks are in poor condition, making them difficult to access. The lack of report results in less attention by the government, therefore, decreasing maintenance. Lack of a platform, namely to effectively report location of sidewalk damages is the main reason reports are not prevalent. Manual reporting, which is still done today, delays the notification of sidewalk damage.</p>
<p align="justify">TrotoTrack is an application designed to facilitate the public in reporting sidewalk damage. Equipped with a Machine Learning-based sidewalk damage detection system, and integrated with Google Maps to accurately display the damaged location. TrotoTrack facilitates sustainable urban development by fostering collaboration between citizens and authorities in enhancing pedestrian facilities.</p>


## Dataset <a name="dataset"></a>
- **Source:** [Dataset Link](https://drive.google.com/drive/folders/1Sd9CN9NAJP24qQrr1IvDX86cCOcHUoxy?usp=sharing) ![DatasetSample][image0]
- **Description:** The model was trained with a dataset of 1,514 sidewalk images collected from Google Maps.
- **Preprocessing:** The images were resized to the corresponding pixels expected by the model and normalized to have pixel values between 0 and 1. This process uses the ImageDataGenerator from the TensorFlow library.

## Model Architecture <a name="model-architecture"></a>

Describe the architecture of the model you are using. For example:

- **Type:** Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), etc.
- **Layers:** Describe the layers and their configurations
- **Hyperparameters:** Learning rate, batch size, epochs, etc.

## Installation <a name="installation"></a>

To install and set up this project, you can use the following code:

```bash
git clone https://github.com/TrotoTrackApp/TrotoTrack-Machine-Learning.git
pip install -r requirements.txt
```

## Training <a name="training"></a>
### Transfer Learning with InceptionV3
![Accuracy][image1]
![losses][image2]
![CM Test][image3]
