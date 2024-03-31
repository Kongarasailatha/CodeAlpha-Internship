# Credit Scoring Model-using-Machine-Learning
 The Python code for building a credit scoring model using Logistic Regression. 
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
1) Efficiency: Automating the loan approval process, saving time and resources for lenders.
2) Consistency: Providing a standardized approach to credit assessment, reducing bias and subjective decision-making.
3) Risk Management: Helping lenders manage risk by identifying high-risk applicants and adjusting loan terms accordingly.
4) Access to Credit: Facilitating access to credit for individuals with limited credit history or unconventional financial backgrounds.


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
### PythonCodeOfAlgorithm.py
This is the file which consist of dataset and there are various differnt algorithms used for training of our model which are as follows:

1) Decision Tree
2) Random Forest
3) Naive Bayes These Three algorithms is used to train our model and all gives an accuracy of over 90
### Explanation of Files
#### Training.csv
->This is the main dataset which has been used in this project. This dataset consist of mainly two columns "Disease" and "Symptoms" but this dataset is preprocessed so it helps in easily clasifying the data. This dataset is used to train our model.
#### Testing.csv
->This is the dataset which has been used to test our model so that we can know the accuracy of our model. this dataset is predefined with output.
### Working with GUI
#### Step 1:
Enter the name in the provided space infront of the label as "Name of the Patient". It is the mandatory field which user have to enter in order to get result.
#### Step-2:
Select 5 Symptoms from the dropdown menu which are labelled as Symptom 1, Symptom 2, Symptom 3, Symptom 4, Symptom 5 respectively. If user is not aware of 5 symptoms then it is mandatory for him to enter atleast 2 starting systems, otherwise the result will not come and a message box will pop up for the same
#### Step-3:
As per user interest,he/she can predict the disease using different algorithms such as Decision tree algorithm, Random forest algorithm, Naive bayes algorithm and K-Nearest neighbour. According to algorithm click on buttons:
Press Prediction 1 for Decision tree algorithm
Press Prediction 2 for Random forest algorithm
Press Prediction 3 for Naive bayes algorithm
(User can predict the disease using more than one algorithm at a time)
#### Step-4:
Disease Recommendation will be available infront of the labels of algorithm of user's choice.
### A picture of GUI Interface
![disease img](https://github.com/Kongarasailatha/CodeAlpha-Internship/assets/140708197/4edf073d-a24f-49f7-b530-98de1075feb6)

#  Handwritten Character Recognition
#### About the Project:
Developed a machine learning model for recognizing handwritten characters.
Implemented a Convolutional Neural Network (CNN) architecture for improved accuracy.
#### Key Achievements:

1) Achieved over 95% accuracy on the test dataset.
2)Enhanced user interface for seamless user interaction.
#### Features
Here are some potential features you could include in your handwritten character recognition project:

1. Model Training: Train a machine learning model using the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits.

2. Convolutional Neural Network (CNN): Implement a CNN architecture for recognizing handwritten characters. CNNs are particularly effective for image recognition tasks.

3. Model Evaluation: Evaluate the trained model's performance using metrics such as accuracy, precision, recall, and F1 score.

4. Deployment: Deploy the trained model using Flask, a lightweight web application framework, to create a web interface for users to interact with.

5. Web Interface: Create a user-friendly web interface where users can input handwritten characters and see the model's predictions.

6. Data Augmentation: Implement data augmentation techniques to increase the diversity of the training dataset, which can improve the model's performance.

7. Hyperparameter Tuning: Experiment with different hyperparameters (e.g., learning rate, batch size, number of layers) to optimize the model's performance.

8. Error Analysis: Conduct error analysis to identify patterns in the model's mistakes and improve its accuracy.

9. Model Interpretability: Explore techniques to make the model more interpretable, such as visualizing the activations of different layers.

These features can help you develop a robust and effective handwritten character recognition system.
ScreenShots of Handwritten Recoginition:
![handwritten img1](https://github.com/Kongarasailatha/CodeAlpha-Internship/assets/140708197/9572316e-31e0-4b2e-84b6-a0b4f1387ff5)
![handwritten img2](https://github.com/Kongarasailatha/CodeAlpha-Internship/assets/140708197/58ce69c6-d7d1-4d96-b44a-132b885e85af)
![handwritten img 3](https://github.com/Kongarasailatha/CodeAlpha-Internship/assets/140708197/aafe9fa7-8076-432d-8b8e-14528cee285e)




