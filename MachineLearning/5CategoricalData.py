# Import libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Create feature
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

# Create one-hot encoder
one_hot = LabelBinarizer()

# One-hot encode feature
print(feature)
print(one_hot.fit_transform(feature))

# View feature classes
print(one_hot.classes_)

# Reverse one-hot encoding
print(one_hot.inverse_transform(one_hot.transform(feature)))

# Import library

print('=======================================================')

# Create dummy variables from feature
print(pd.get_dummies(feature[:, 0]))

print('=======================================================')

# Create multiclass feature
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()

# One-hot encode multiclass feature
one_hot_multiclass.fit_transform(multiclass_feature)

# View classes
print(one_hot_multiclass.classes_)

print('==========================REPLACE=============================')

# Load library

# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

# Create mapper
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}

# Replace feature values with scale
dataframe["Score"].replace(scale_mapper)

dataframe = pd.DataFrame({"Score": ["Low",
                                    "Low",
                                    "Medium",
                                    "Medium",
                                    "High",
                                    "Barely More Than Medium"]})

scale_mapper = {"Low": 1,
                "Medium": 2,
                "Barely More Than Medium": 3,
                "High": 4}

dataframe["Score"].replace(scale_mapper)

scale_mapper = {"Low": 1,
                "Medium": 2,
                "Barely More Than Medium": 2.1,
                "High": 3}

dataframe["Score"].replace(scale_mapper)

print('=========================Feature Dict==============================')

# Import library

# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]

# Create dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)

# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)

# View feature matrix
print(features)

# Get feature names
get_feature_names_out = dictvectorizer.get_feature_names_out()

# View feature names
print(get_feature_names_out)

# Import library

print('===========================pd ver============================')

# Create dataframe from features
print(pd.DataFrame(features, columns=get_feature_names_out))

print('===========================dictvectorizer============================')

# Create word counts dictionaries for four documents
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

# Create list
doc_word_counts = [doc_1_word_count,
                   doc_2_word_count,
                   doc_3_word_count,
                   doc_4_word_count]

# Convert list of word count dictionaries into feature matrix
print(dictvectorizer.fit_transform(doc_word_counts))

print('=======================|Missing class vals|=======================')
print('=======================|predict missing vals|=======================')

# Load libraries

# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:, 1:], X[:, 0])

# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:, 1:])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))

# Join two feature matrices
print(X)
print()
print(X_with_nan)
print()
print(clf)
print()
print(trained_model)
print()
print(imputed_values)
print()
print(X_with_imputed)
print()
print(np.vstack((X_with_imputed, X)))

print('=======================|fill w most_frequent (mode) val|=======================')

# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))

imputer = SimpleImputer(strategy='most_frequent')

print(X_complete)
print('correct')
print(imputer.fit_transform(X_complete))

print('=======================|imbalanced cases|=======================')

# Load libraries

# Load iris data
iris = load_iris()

# Create feature matrix
features = iris.data

# Create target vector
target = iris.target

#print('feat: ', features)
#rint('targ: ',target)

# Remove first 40 observations
features = features[40:, :]
target = target[40:]

# Create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)

"""
three balanced classes of 50 observations, each indicating the species of flower (Iris setosa, Iris virginica, and Iris versicolor). To unbalance the dataset, we remove 40 of the 50 Iris setosa observations and then merge the Iris virginica and Iris versicolor classes. The end result is a binary target vector indicating if an observation is an Iris setosa flower or not. The result is 10 observations of Iris setosa (class 0) and 100 observations of not Iris setosa (class 1):
"""

# Look at the imbalanced target vector
print('2feat: ', features)
print('2targ: ', target)

print('====================================================')

# Create weights
weights = {0: .9, 1: 0.1}

# Create random forest classifier with weights
RandomForestClassifier(class_weight=weights)

# Train a random forest with balanced class weights
RandomForestClassifier(class_weight="balanced")

print('====================================================')
#ALTERNATIVE downsample majority or upsample minority

# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# Number of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# For every observation of class 0, randomly sample
# from class 1 without replacement
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# Join together class 0's target vector with the
# downsampled class 1's target vector
np.hstack((target[i_class0], target[i_class1_downsampled]))

# Join together class 0's feature matrix with the
# downsampled class 1's feature matrix
np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5]

print('====================================================')
#upsample minority. reverse downsampling

# For every observation in class 1, randomly sample from class 0 with replacement
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((target[i_class0_upsampled], target[i_class1]))

# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5]