'''
**********
Your First Machine Learning Project in Python Step-By-Step
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
Date: 19/04/2019
*********
''' 


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-lenght', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)



# Dimensions of Dataset
# shape -> how many instances (row) and how many attributes (columns) the data contains with shape property
print(dataset.shape)



# Peek at the Data
# head -> you should see the first 20 rows of the data
print(dataset.head(20))



# Statistical Summary
# descriptions -> Now we can take a look at a summary of each attribute
# We can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters
print(dataset.describe())



# Class Distribution
# class distribution -> Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count
print(dataset.groupby('class').size())



# Data Visualization(two types of plots)

# 1. Univariate Plots(plots to better understand each attribute)
# Given that the input variables are numeric, we can create box and whisker plots of each
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# We can also create a histogram of each input variable to get an idea of the distribution
# histograms
dataset.hist()
plt.show()


# 2. Multivariate Plots(plots to better understand the relationships between attributes)
# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables
# scatter plot matrix
scatter_matrix(dataset)
plt.show()



# Evaluate Some Algorithms
# Now it is time to create some models of the data and estimate their accuracy on unseen data

# 1. Create a Validation Dataset
# We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# 2. Test Harness
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# 3. Build Models

# Let’s evaluate 6 different algorithms:
# - Logistic Regression (LR)
# - Linear Discriminant Analysis (LDA)
# - K-Nearest Neighbors (KNN).
# - Classification and Regression Trees (CART).
# - Gaussian Naive Bayes (NB).
# - Support Vector Machines (SVM).
# Simple linear (LR and LDA) X Nonlinear (KNN, CART, NB and SVM) 

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# 4. Select Best Model

# We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate
# In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score.
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show() # You can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy



# Make Predictions
# The KNN algorithm is very simple and was an accurate model based on our tests. Now we want to get an idea of the accuracy of the model on our validation set
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# We can see that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small)

