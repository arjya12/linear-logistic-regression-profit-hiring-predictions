import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.
Hint: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 
"""

# Open the csv file RegressionData.csv in Excel, notepad++ or any other applications to have a 
# rough overview of the data at hand. 
# You will notice that there are several instances (rows), of 2 features (columns). 
# The values to be predicted are reported in the 2nd column.

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'X' and the second feature 'y' (these are the labels)
data = pandas.read_csv('RegressionData.csv', header = None, names=['X', 'y']) # 5 points
# Reshape the data so that it can be processed properly
X = data['X'].values.reshape(-1,1) # 5 points
y = data['y'] # 5 points
# Plot the data using a scatter plot to visualize the data
plt.scatter(X, y) # 5 points

# Linear regression using least squares optimization
reg = linear_model.LinearRegression() # 5 points
reg.fit(X, y) # 5 points

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) # 5 points
plt.scatter(X,y, c='b') # 5 points
plt.plot(X, y_pred, 'r') # 5 points
fig.canvas.draw()

# # Complete the following print statement (replace the blanks _____ by using a command, do not hard-code the values):
print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_[0])
# 8 points

# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]])[0])
# 8 points
    
"""
PART 2: logistic regression 
You are a recruiter and your goal is to predict whether an applicant is likely to get hired or rejected. 
You have gathered data over the years that you intend to use as a training set. 
Your task is to use logistic regression to build a model that predicts whether an applicant is likely to
be hired or not, based on the results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
"""

# Open the csv file in Excel, notepad++ or any other applications to have a rough overview of the data at hand. 

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv('LogisticRegressionData.csv', header = None, names=['Score1', 'Score2', 'y']) # 2 points

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[['Score1', 'Score2']] # 2 points
y = data['y'] # 2 points

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) # 2 points
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression() # 2 points
regS.fit(X, y) # 2 points

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X) # 2 points
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[y_pred[i]], c = c[y_pred[i]]) # 2 points
fig.canvas.draw()
# Notice that some of the training instances are not correctly classified. These are the training errors.

"""
PART 3: Multi-class classification using logistic regression 
Not all classification algorithms can support multi-class classification (classification tasks with more than two classes).
Logistic Regression was designed for binary classification.
One approach to alleviate this shortcoming, is to split the dataset into multiple binary classification datasets 
and fit a binary classification model on each. 
Two different examples of this approach are the One-vs-Rest and One-vs-One strategies.
"""

#  One-vs-Rest method (a.k.a. One-vs-All)

# Explain below how the One-vs-Rest method works for multi-class classification # 12 points
"""
The One-vs-Rest (OvR) method, also known as One-vs-All (OvA), is a strategy used to extend binary classification algorithms, such as logistic regression, to multi-class classification problems. In this approach, for a dataset with N classes, N separate binary classifiers are trained. Each classifier is designed to distinguish one specific class from all the other classes combined. For example, if there are three classes (Class A, Class B, and Class C), the method trains three classifiers: one to distinguish Class A from Class B and Class C, another to distinguish Class B from Class A and Class C, and a third to distinguish Class C from Class A and Class B.

During training, each classifier learns to predict the probability that a given sample belongs to its designated class versus any other class. When making predictions, each classifier outputs a probability score for its respective class. The class with the highest probability score across all classifiers is selected as the final predicted class for the sample. This method is simple and efficient, as it only requires training N classifiers. However, it can struggle with imbalanced datasets or when classes overlap significantly, as some classifiers may become biased toward the majority class.
"""

# Explain below how the One-Vs-One method works for multi-class classification # 11 points
"""
The One-vs-One (OvO) method is another strategy for multi-class classification. Unlike the One-vs-Rest approach, which compares one class against all others, the One-vs-One method compares every pair of classes individually. For a dataset with N classes, the OvO method trains N * (N-1)/2 binary classifiers. Each classifier is trained to distinguish between one pair of classes. For example, if there are three classes (Class A, Class B, and Class C), the method trains three classifiers: one to distinguish Class A from Class B, another to distinguish Class A from Class C, and a third to distinguish Class B from Class C.

During training, each classifier is trained on a subset of the data that contains only the two classes it is distinguishing between. When making predictions, each classifier votes for one of the two classes it was trained on. The class that receives the most votes across all classifiers is selected as the final predicted class for the sample. This method can handle overlapping classes better than One-vs-Rest because each classifier focuses on only two classes. It also works well with small datasets, as each classifier is trained on a smaller subset of the data. However, it is computationally more expensive than One-vs-Rest, as it requires training N * (N-1)/2 classifiers, and managing the results of many classifiers can be more complex.
"""