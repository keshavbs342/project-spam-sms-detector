# 📱 SMS Spam Detection Pipeline

## 🎯 Project Overview
This repository contains a robust Natural Language Processing (NLP) machine learning pipeline designed to automatically classify SMS messages as either **Spam** (unwanted/promotional) or **Ham** (normal/safe). By leveraging supervised learning algorithms, the system provides a computational defense against unwanted messaging, showcasing practical applications of text classification and predictive modeling.

## 🛠️ Technology Stack
* **Language:** Python
* **Data Manipulation & Cleaning:** Pandas
* **Machine Learning Framework:** Scikit-Learn
* **Algorithm:** Multinomial Naive Bayes (MultinomialNB)
* **Text Processing:** TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency)

## 🧠 System Architecture & Workflow
1. **Data Preprocessing:** The raw SMS dataset is cleaned, extracting the relevant text data and converting string labels into binary numerical targets (`0` for Ham, `1` for Spam).
2. **Feature Extraction:** Using `TfidfVectorizer`, standard English text is transformed into mathematical matrices. This process evaluates word frequency while filtering out uninformative standard English "stop words".
3. **Model Training:** A Naive Bayes classifier is trained on the vectorized training data to recognize probabilistic patterns and trigger words inherent in spam vocabulary.
4. **Prediction & Evaluation:** The trained model analyzes unseen testing data to predict classifications based on learned historical patterns.

## 📊 Performance Metrics
* **Accuracy:** The model achieved a highly accurate classification score of **96.68%** on the unseen testing dataset, demonstrating strong precision in distinguishing between legitimate and promotional messages.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed on your local machine. You will need to install the following dependencies via your terminal:
```bash
pip install pandas scikit-learn
```

### Running the Project Locally

1. Clone this repository to your local environment.
2. Ensure the `spam.csv` dataset is located in the root directory alongside the main script.
3. Execute the Python script:

```bash
python main.py

```

---

## 👨‍💻 Author

**Keshav Shukla**

*B.Tech in Electronics and Communication Engineering | Shri Ramdeobaba College of Engineering and Management*

* 📧 Email: keshavbs342@gmail.com
* 💡 Focus: Python, Natural Language Processing, and Machine Learning
