# BERT-Based Email Classification with Imbalance Handling

This project implements a **Binary Classification Model** using the **BERT (Bidirectional Encoder Representations from Transformers)** architecture to classify emails into spam and non-spam categories. The focus of the project is on **handling class imbalance** in the dataset and improving the model's ability to generalize on such data. 

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Techniques Used](#techniques-used)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Overview

The goal of this project is to classify emails into two categories: spam and non-spam. We use **BERT** for feature extraction and train a deep learning model on these features to perform the classification task. Given that real-world email datasets often suffer from class imbalance (i.e., more non-spam emails than spam), the project incorporates techniques to handle this imbalance.

### Key Features:
- **BERT Transformer Model**: Used for email feature extraction.
- **Imbalance Handling**: Techniques like oversampling, undersampling, or class-weighting are employed to balance the training data.
- **Binary Classification**: The model outputs a probability of whether an email is spam or not.
  
## Model Architecture

The model uses the **BERT pre-trained language model** for text feature extraction. The core architecture involves the following steps:

1. **Input Text**: Email text is inputted as a string.
2. **Preprocessing with BERT**: The text is tokenized and preprocessed using BERT tokenizer.
3. **BERT Encoder**: The preprocessed text is passed through the BERT encoder to extract rich contextual embeddings.
4. **Dropout Layer**: A dropout layer with a dropout rate of 0.1 is added to reduce overfitting.
5. **Dense Layer**: A dense layer with a sigmoid activation function is used for binary classification.

The model is implemented using the **Functional API** in TensorFlow/Keras for flexibility and extensibility.

```python
# Example Architecture in Code
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Construct the final model
model = tf.keras.Model(inputs=[text_input], outputs=[l])
```

## Techniques Used

### 1. **Imbalance Handling**
- **Class Weights**: Adjusting weights for each class to handle imbalanced data.
- **Oversampling/Undersampling**: Techniques to balance the dataset by duplicating minority class samples or reducing majority class samples.

### 2. **BERT for Text Embeddings**
- Pre-trained **BERT Base** model from TensorFlow Hub is used to generate embeddings for emails, capturing rich language patterns.
- <img width="800" alt="image" src="https://github.com/user-attachments/assets/ec105a44-6601-4857-9eb3-090f50181992">


### 3. **Sigmoid Activation Function**
- The final layer uses a **sigmoid activation function** for binary classification, giving a probability score for each class.

### 4. **Dropout**
- A **dropout layer** with a 0.1 rate is used to prevent overfitting during training.

## Dataset

The dataset contains emails labeled as **spam** or **non-spam**. Due to the nature of email datasets, class imbalance is a significant concern, and techniques to mitigate this are employed in the project.

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

These metrics help to evaluate not only the overall accuracy but also how well the model is performing on the minority (spam) class.

## Results
<img width="371" alt="image" src="https://github.com/user-attachments/assets/922a6e20-3323-4e01-ac4a-033972da9fe4">


On the test set, the model achieved the following results:

```
              precision    recall  f1-score   support

           0       0.99      0.82      0.90       187
           1       0.85      0.99      0.91       187

    accuracy                           0.91       374
   macro avg       0.92      0.91      0.91       374
weighted avg       0.92      0.91      0.91       374
```

The model shows strong performance in both precision and recall, especially for the minority class.

## Installation

To run this project locally, clone the repository and install the required packages.

```bash
git clone https://github.com/yourusername/BERT_email_classification.git
cd BERT_email_classification
pip install -r requirements.txt
```

## Usage

1. **Preprocessing**: The email data needs to be tokenized using the BERT tokenizer.
2. **Training**: The model is trained on the preprocessed text, with special attention to imbalance handling.
3. **Prediction**: Use the trained model to classify new emails as spam or non-spam.

```python
reviews = [
    'Enter a chance to win $5000, hurry up!',
    'Hey, are you free for lunch tomorrow?'
]

predictions = model.predict(reviews)
```
