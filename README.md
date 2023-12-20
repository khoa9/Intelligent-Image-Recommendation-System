# Intelligent Images Classification using Convolutional Neural Network

## Overview

This project aims to classify pictures into three categories: river, mountain, and city using a Convolutional Neural Network (CNN). The project is divided into two main parts: data collection and model building. The dataset is obtained from Flickr API, ensuring that only natural scenes are included, excluding human elements. The models used for classification are a 14-layer CNN and a hybrid model that employs VGG16 for preprocessing followed by a 7-layer CNN.

### Data colelction
We have an R Script that will show the full process of making API call to Flickr to receive all of the required pictures for each scenes we need. Next we put all of the pictures into one folder and upload it on Google Drive. You can download the folder from [here]:(https://drive.google.com/file/d/1xQI78UG-BqojbXWs8K0F1xI6KEWkYPDS/view?usp=drive_link)

### Project Structure
The project is structured as follows:

**1. Data Collection**: The dataset was collected from the Flickr API through a set of code which can be viewed in Project-Group1-FlickrDataCollection.R file

**2. Data Preprocessing**: Data preprocessing involves selecting the pictures that meets most of our critera including no human in pictures, clear pictures without bubble and each scense picture must be completely seperated from each other.

**3. Data Storage**: Store nearly 1,700 pictures of three different categories to serve the model building purpose.
 
**5. Model Selection**: We experiment with different neural network techniques like convolutional neural network (CNN) and we also leverage the pre-train CNN model - VGG16 as a function to process and extract data features.

**6. Model Evaluation**: The model's performance is evaluated using accuracy metric and whether that model is overfitting or not.

**7. Results**: We present the results of the trained model, including insights of which model will perform the best and we also test the best model with an RDS file containing test data.

**8. Conclusion**: We summarize the findings of the project, discuss potential use cases, and suggest areas for further research. Everything is included in the Rmd file of this project

### Steps to conduct this project on your local machine:
To ensure that the project works fine on your end, here are the steps:

**1.** Download all the files in the submissions

**2.** Download the data zip file from Google Drive: https://drive.google.com/file/d/1xQI78UG-BqojbXWs8K0F1xI6KEWkYPDS/view?usp=drive_link

**3.** Put all the submitted files and the downloaded zip data file within the same directory

**4.** Execute the R Script - Project-Group1-Modelling.R

### Results
The project aims to have a good accuracy on classifying different natural scenes pictures. As a result, our best model delivered the accuracy of around 83% which the low chance of being overfitting. 

### Future Work
The trained models can be used for classifying new images into the specified categories. You can further integrate this classification system into recommendation systems or other image classification models.

With more data and time to test different ways to enhance the model, we believe the accuracy of this model will be much more higher.

Feel free to customize and extend the code to suit your specific requirements.

### Note
Please ensure that you have the required R packages and dependencies installed before running the scripts. Also, please adhere to Flickr API usage policies and respect the terms of service while collecting data from the platform.
