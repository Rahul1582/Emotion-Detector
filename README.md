# Emotion-detection Using CNN

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks.
The model is trained on the **FER-2013** dataset.

## Dependencies

To install the required packages, run `pip install -r requirements.txt`.

## Technologies Used
```
1.Python

2.Convolution Neural Network(CNN)

3.Open CV

4.Data Augmentation
```

## To Run

First, clone the repository and enter the folder src

Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing) and unzip it inside the `src` folder. This will create the folder `data`.

I had added the dataset into gitignore as it is a very big file.

If you want to train this model, use:  
cd src
python emotions.py --run train


If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing).

If you want to use the web camera or give input as a video to detect emotions just run-
cd src
python emotions.py --run test
with USE_WEBCAM as True or False

If you want to detect emotions in a picture just run-
cd src
python emotions.py --run picture

With a simple 4-layer CNN, the test accuracy reached 63% in 50 epochs.

 
## Data Preparation (optional)

The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file.

### Demo

Model Summary--
<img src="images/summary.PNG" width=700 height=600>
<br><br>

<img src ="images/happy.PNG"  width=700 height=500>
<br>

<img src="images/surprised.PNG" width=700 height=500>
<br>



