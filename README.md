# CS506 Midterm

## Overview

Khoa Cao (Kaggle: *kcao23*)

This repository contains my work for the CS506 midterm project, where I developed a machine learning model to predict album review scores based on various features extracted and engineered from the provided dataset using Machine Learning and NLP techniques.


1. Data Exploration

My data exploration steps are documented in the `eda.ipynb` notebook.

First, I examined the quality of the data, looking missing values and checking data types. I also visualized the distributions of key features and `Score` to understand their characteristics. I noticed that some features had skewed distributions, which informed my feature engineering decisions later. Additionally, I noted that `Score` was heavily imbalanced, concentrated around 1 and 5.

2. Feature Extraction and Engineering

In the `feature_engineering.ipynb` notebook, I focused on creating new features and transforming existing ones to improve model performance. I experiemented with grouping genres into broader categories and one-hot encoding them. For skewed numerical features, I applied log transformations to reduce their skewness. 

For the text features (`reviewText` and `summary`), I used TF-IDF vectorization to convert them into numerical representations, reducing their dimensionality with Truncated SVD. I also created additional features such as review length, summary length, and sentiment scores using the VADER sentiment analysis tool.

3. Model Creation and Assumptions

Before feeding data into the model, I scaled the numerical features using StandardScaler to ensure they were on a similar scale.

I implemented a Logistic Regression model to predict the `Score` based on the engineered features. I split the data into training and validation sets to evaluate model performance. To address class impbalance, I utilized a Balanced Bagging Classifier, which combines multiple Logistic Regression models trained on balanced subsets of the data.

4. Model Tuning

I intended to use GridSearch CV to tune hyperparameters (e.g. C, max_iter, solver) for a Logistic Regression model. However, due to time constraints, I was unable to complete this step.

5. Model Evaluation and Performance

I evaluated the model using accuracy, precision, recall, and F1-score metrics. The model achieved an macro F1-score of approximately 0.48 on the validation set, indicating moderate performance. The confusion matrix revealed that the model struggled to accurately predict mid-range scores (2, 3, 4), often misclassifying them as 1 or 5.

6. Struggles / Issues / Challenges

One of the main challenges I faced was handling the high dimensionality of the text features after TF-IDF vectorization. I mitigated this by applying Truncated SVD to reduce the number of features while retaining important information. Additionally, I was limited by time constraints and the computational resources available, which affected the depth of hyperparameter tuning, model experimentation, and model complexity I could explore.

I also realized last-minute that I introduced data leakage by creating features, specifically the genre one-hot encodings and TF-IDF features, using the entire dataset before splitting into training and validation sets. This likely inflated the model's performance metrics. In a production scenario, I would ensure that feature engineering is performed only on the training set to prevent leakage.

I also struggled with feature engineering decisions, particularly around how to best represent the text data and which numerical features to include. More experimentation and validation would be needed to optimize these choices.
