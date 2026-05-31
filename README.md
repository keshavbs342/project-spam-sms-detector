# 📱 SMS Spam Detection using Machine Learning

## Overview

This project is a Natural Language Processing (NLP) machine learning model built in Python. It automatically reads text messages and classifies them as either "Spam" (unwanted) or "Ham" (safe).

## 🛠️ Technologies Used

- **Python**: The core programming language.
- **Pandas**: Used for data manipulation and cleaning.
- **Scikit-Learn**: Used for the machine learning pipeline (Vectorization and Model Training).

## 🧠 How It Works

1. **Data Preprocessing**: The raw SMS data is cleaned, and text labels are converted into binary numbers.
2. **Feature Extraction**: Using `TfidfVectorizer`, English text is converted into numerical data, while common "stop words" are filtered out.
3. **Model Training**: A Multinomial Naive Bayes classifier is trained on the data to recognize patterns in spam vocabulary.
4. **Evaluation**: The model achieves an accuracy of ~97% on unseen test data.

## 🚀 How to Run This Project

1. Clone this repository to your local machine.
2. Ensure you have Python installed, along with Pandas and Scikit-Learn.
3. Run `python main.py` in your terminal.
