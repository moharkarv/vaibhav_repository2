# Natural Language Processing with Disaster Tweets

This project aims to classify tweets as disaster-related or not using NLP techniques and machine learning. Developed by a team of six members, this project demonstrates the power of text preprocessing, feature extraction, and model building to solve real-world problems.

## Table of Contents
- About the Project
- Dataset
- Setup and Installation
- Project Workflow
- Team Members
- Acknowledgments

**1. About the Project**

The project leverages Twitter data to determine whether a tweet is related to a disaster. It utilizes state-of-the-art NLP techniques such as stemming, tokenization, and Bag of Words (BoW) for feature extraction, followed by machine learning algorithms for classification.

**2. Dataset**

The dataset is sourced from the [Kaggle Competition - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started). It includes tweet texts, keywords, and labels for disaster-related or non-disaster tweets.

**3. Setup and Installation**
   
3.1 Clone the repository:
   git clone https://github.com/your-repo-url.git


3.2 Install the required dependencies:
  pip install -r requirements.txt


3.3 Run the preprocessing or training scripts:
  disaster_classification_project.py



**4. Project Workflow**

**Data Aquisition**: Collection of The Data   
**Data Preprocessing**: Cleaned text data using techniques such as tokenization and stemming.
**Feature Engineering**: Implemented Bag of Words and TF-IDF for feature extraction.
**Model Training**: Trained classifiers such as Logistic Regression and Random Forest.
**Evaluation**: Evaluated models using metrics like accuracy, precision, and recall.


**5. Team Members**

1. Data Collection
[Team Member Name 1]:

Collected disaster-related datasets from sources like [e.g., Kaggle, Twitter, or other platforms].
Verified the integrity and relevance of the data for the classification task.

3. Data Preprocessing
[Team Member Name 2]: Ravichandra

Handled missing values and outliers in the dataset.
Applied text preprocessing, including tokenization, stemming, lemmatization, and stopword removal.
Normalized data for consistency.

5. Exploratory Data Analysis (EDA)
[Team Member Name 3]:

Conducted statistical analysis to identify trends in disaster types.
Created visualizations (e.g., bar charts, word clouds) to highlight key patterns in the data.
Analyzed correlations between features and target labels.


6. Feature Engineering
[Team Member Name 4]: _Abhishek Gaikwad_

Extracted text-based features using methods such as TF-IDF, Word2Vec, and GloVe embeddings.
Added sentiment scores and disaster-related keyword frequency as features.
Optimized feature selection to enhance model performance.


7. Model Development
[Team Member Name 5]:

Implemented machine learning algorithms like Logistic Regression, Naive Bayes, and Random Forest.
Conducted hyperparameter tuning and cross-validation.
Evaluated models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.


6.**Acknowledgments**

Kaggle for the dataset.
Open-source libraries such as NLTK, scikit-learn, and pandas.






