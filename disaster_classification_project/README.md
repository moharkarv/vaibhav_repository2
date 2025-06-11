This project aims to classify tweets as disaster-related or not using NLP techniques and machine learning. Developed by a team of six members, this project demonstrates the power of text preprocessing, feature extraction, and model building to solve real-world problems.



1. Data Collection
[Bhavana Warghane]:

Collected disaster-related data from Kaggle.
Imported the dataset into Jupyter Notebook for further processing.
Hnadle Missing Values
Check Duplicate rows.

3. Data Preprocessing
[Ravichandra]

Applied text preprocessing, including tokenization, stemming, lemmatization, and stopword removal.

[Vaibhav Moharkar]:

Split the dataset into training and testing sets using an 80-20 split with stratification to ensure balanced class distribution.
Applied SMOTE (Synthetic Minority Oversampling Technique) on the training data to handle class imbalance.

4. Feature Engineering
[Abhishek Gaikwad]

Extracted text-based features using methods such as TF-IDF, Word2Vec.
Optimized feature selection to enhance model performance.


5. Model Development
[Pranita Lokhande]

Implemented machine learning algorithms like XG Boost,SVM and Random Forest. Conducted hyperparameter tuning and cross-validation. Evaluated models using metrics such as accuracy, precision, recall, F1-score.

[Rutuja Kandhare..]

Implemented machine learning algorithms like Logistic Regression, Naive Bayes Conducted hyperparameter tuning and cross-validation. Evaluated models using metrics such as accuracy, precision, recall, F1-score.






Results Summary - TF-IDF

| **Model**                                  | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Confusion Matrix**                |
|--------------------------------------------|--------------|---------------|------------|--------------|-------------------------------------|
| Gaussian Naive Bayes                       | 0.7623       | 0.7704        | 0.6361     | 0.6968       | [[745, 124], [238, 416]]            |
| Multinomial Naive Bayes (MNB)              | 0.8207       | 0.8408        | 0.7187     | 0.7749       | [[780, 89], [184, 470]]             |
| Bernoulli Naive Bayes (BNB)                | 0.8162       | 0.8979        | 0.6453     | 0.7509       | [[821, 48], [232, 422]]             |
| Logistic Regression                        | 0.8181       | 0.7969        | 0.7737     | 0.7851       | -                                   |
| Support Vector Classifier (SVC)            | 0.7951       | 0.7749        | 0.7370     | -            | -                                   |
| K-Nearest Neighbors (KN)                   | 0.6461       | 0.5842        | 0.6101     | -            | -                                   |
| Decision Tree (DT)                         | 0.6415       | 0.8214        | 0.2110     | -            | -                                   |
| Random Forest (RF)                         | 0.7978       | 0.8046        | 0.6988     | -            | -                                   |
| AdaBoost                                   | 0.7590       | 0.7682        | 0.6284     | -            | -                                   |
| Bagging Classifier (BgC)                   | 0.7781       | 0.7380        | 0.7492     | -            | -                                   |
| Extra Trees Classifier (ETC)               | 0.7879       | 0.7763        | 0.7110     | -            | -                                   |
| Gradient Boosting Decision Trees (GBDT)    | 0.7518       | 0.8194        | 0.5413     | -            | -                                   |
| XGBoost (xgb)                              | 0.7833       | 0.8000        | 0.6606     | -            | -                                   |






Results Summary - Word2Vec

Model                                  | Accuracy | Precision | Recall   | F1-Score | Confusion Matrix
---------------------------------------|----------|-----------|----------|----------|-----------------------------
Gaussian Naive Bayes (GNB)             | 0.4609   | 0.4425    | 0.9817   | 0.6100   | [[60, 809], [12, 642]]
Multinomial Naive Bayes (MNB)          | Error    | -         | -        | -        | -
Bernoulli Naive Bayes (BNB)            | 0.6054   | 0.5551    | 0.4083   | 0.4705   | [[655, 214], [387, 267]]
Logistic Regression (LR)               | 0.6205   | 0.5651    | 0.5046   | 0.5331   | -
Support Vector Classifier (SVC)        | 0.4498   | 0.3763    | 0.4281   | -        | -
K-Nearest Neighbors (KN)               | 0.6021   | 0.5275    | 0.7049   | -        | -
Decision Tree (DT)                     | 0.6467   | 0.5895    | 0.5841   | -        | -
Random Forest (RF)                     | 0.7039   | 0.6765    | 0.5948   | -        | -
AdaBoost                               | 0.6599   | 0.5971    | 0.6391   | -        | -
Bagging Classifier (BgC)               | 0.6993   | 0.6617    | 0.6131   | -        | -
Extra Trees Classifier (ETC)           | 0.6947   | 0.6760    | 0.5550   | -        | -
Gradient Boosting Decision Trees (GBDT)| 0.6861   | 0.6392    | 0.6177   | -        | -
XGBoost (xgb)                          | 0.7144   | 0.6698    | 0.6606   | -        | -







Logistic Regression on TF-IDF values:
Accuracy: 0.8181
Precision: 0.7969
Recall: 0.7737
F1-Score: 0.7851

Logistic regression provides better results compared to other models; that's why we have selected it for model development.
