# Twitter Sentiment Analysis

## Overview
This project aims to classify Twitter comments based on their sentiment (Positive, Negative, or Neutral). It leverages Natural Language Processing (NLP) techniques and Machine Learning algorithms to analyze and predict sentiment from text data.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

## Data Collection
The dataset consists of Twitter comments collected using the Twitter API. Each comment is labeled with its corresponding sentiment category (positive, negative, or neutral).

## Data Preprocessing
The following preprocessing steps were performed:
- Loading the dataset
- Handling missing values
- Tokenization
- Removing stop words
- Stemming

## Model Training
Multiple machine learning models were trained and evaluated:
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Convolutional Neural Network (CNN)**

These models were assessed based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## Model Evaluation
For each model, confusion matrices and classification reports were generated to evaluate their performance in classifying tweet sentiments.

## Results
- The **Convolutional Neural Network (CNN)** and **Logistic Regression** models demonstrated the highest accuracy.
- These models were most effective in accurately segregating Twitter comments by sentiment.
