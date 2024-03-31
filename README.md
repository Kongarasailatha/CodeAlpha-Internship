# Credit Scoring Model-using-Machine-Learning
This document explains the Python code for building a credit scoring model using Logistic Regression. 
### Libraries
#### 1) Libraries and Data Loading The code imports necessary libraries: • pandas: Data manipulation 
• numpy: Numerical operations 
• sklearn.model_selection: Train-test split 
• sklearn.preprocessing: Data scaling • sklearn.metrics: Model evaluation metrics 
• sklearn.linear_model: Logistic regression model
• joblib: Saves models for later use It then reads the credit scoring dataset from an Excel file named "a_Dataset_CreditScoring.xlsx" and displays its shape (number of rows and columns) and the first few rows (using head()) for initial exploration.
#### 2) Data Preprocessing
• The code drops the 'ID' column as it likely doesn't contribute to predicting credit risk.
• It checks for missing values using isna().sum(). 
• Missing values are filled with the mean value of each feature in the dataset using fillna(dataset.mean()).
• The target variable (denoting good or bad credit risk) is separated from the features using .iloc.
-> y: Represents the target variable.
->X: Represents the features used for prediction.
3) Train-Test Split The code splits the data into training and testing sets using train_test_split: 
• X_train: Training data features for model training. 
• X_test: Testing data features for model evaluation.
• y_train: Training data target labels for model training. 
• y_test: Testing data target labels for model evaluation. The test size is set to 20% (test_size=0.2) using a random seed (random_state=0) for reproducibility.
4) This can improve the performance of some machine learning models. The scaler is then used to transform the features in the testing set (X_test) using the same parameters learned from the training data.
5) Model Training and Saving • A Logistic Regression classifier is created (classifier). 
• The model is trained on the training data (classifier.fit(X_train, y_train))
• The trained model is saved using joblib.dump for later use.
6) Prediction and Evaluation
-> The model makes predictions on the testing data (y_pred = classifier.predict(X_test)) 
-> The accuracy score is the proportion of predictions that were correct.
7) Generating Prediction Probabilities • The model predicts probabilities of belonging to each class (classifier.predict_proba(X_test)) for the test data. 
• This results in a NumPy array with two columns, likely representing probabilities for good and bad credit risk. 
• These probabilities are stored in a DataFrame named df_prediction_prob.
### Benefits of a financial credit scoring model include:
-> Efficiency: Automating the loan approval process, saving time and resources for lenders.
->Consistency: Providing a standardized approach to credit assessment, reducing bias and subjective decision-making.
->Risk Management: Helping lenders manage risk by identifying high-risk applicants and adjusting loan terms accordingly.
->Access to Credit: Facilitating access to credit for individuals with limited credit history or unconventional financial backgrounds.

# Diseases Prediction By using Machine Learning
 To share my internship project at CodeAlpha: Disease Prediction Task!

##  About the Project:

Developing a machine learning model for disease prediction based on symptoms.
Using a dataset of patient records to train and test the model.
Implementing various machine learning algorithms for prediction.
### Key Objectives:

To accurately predict the likelihood of disease based on symptoms.
To improve early detection and diagnosis of diseases.
To contribute to advancements in healthcare technology.
🔍 Approach:

Exploring different machine learning algorithms such as decision trees, random forests, and neural networks.
Utilizing data preprocessing and feature engineering techniques for model training.
Evaluating model performance using metrics such as accuracy, precision, recall, and F1 score.
🚀 Impact:

Enhancing healthcare services by enabling early intervention and treatment.
Empowering patients with valuable insights into their health.
Contributing to the development of predictive healthcare technologies.
👉 Looking for Opportunities:

Open to collaborations, internships, or research opportunities in the field of healthcare and machine learning.
📩 Reach out: Interested in learning more? Feel free to message me or connect on LinkedIn!
