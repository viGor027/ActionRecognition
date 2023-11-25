# Overview

The project aimed to develop a model capable of recognizing different types of sport activities in videos.

## Tech stack
- tensorflow, keras
- cv2
- numpy
- sklearn
- matplotlib

## Process

The project utilized the MobileNet network as its backbone. The process involved:

1. Exploring optimal learning rates and determining the number of layers and neurons at each layer.
2. Initial model versions were trained, and the confusion matrix was analyzed. To address model errors, I chose to increase the number of frames passed to the model for each sample.
3. Due to memory limitations in a home environment, further increasing the number of frames caused errors. Instead, I implemented a strategy during training to yield more samples of misclassified classes, optimizing learning from these instances.

# Results

0 - Archery  
1 - CleanAndJerk  
2 - Diving  
3 - PushUps  
4 - Skiing   

![confusion matrix](https://i.ibb.co/kXpMrT6/confusion-matrices.png)

## Dataset link

[Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

# How to train the model yourself

First in the directory of cloned repo install all required libraries via:

```pip install -r requirements.txt```

Then download the dataset, and unpack it into ```data``` folder so your folder structure looks as follows:

![Structure](https://i.ibb.co/M60SdHQ/Zrzut-ekranu-2023-11-07-095415.png)

Now you can run ```train.py```
