"""
Repurcutions of misclassifying a fake news as a real news is higher that classifying a real news as fake news.
Since fake news can later be filtered out and then evaluated moderators, in my opinion we should be concerned with misclassifyng fake news.

However from the company's point of view, a misclassification of real news as fake news can lead to banning of a genuine account, which can lead to loss of business for the company.

For the sake of my research, I am going to find a model that is concerned with misclassificaton of fake news as real news. Hence, Precision is going to be my metric of assessment for the model. I will also look at F1 score and accuracy to reduce bias.
"""

# Importing necessary Files
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV




path='C:/Southampton Studies/ML Tech/Coursework/assignment-comp3222-comp6246-mediaeval2015-dataset/assignment-comp3222-comp6246-mediaeval2015-dataset/'

"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""##### Reading the files"""
#The files already have necessary features extracted

df_train = pd.read_csv(path+'training_data_extracted.csv') #Reading training file
df_test = pd.read_csv(path+'test_data_extracted.csv') #Reading testing file


"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""##### subsetting only english data"""
df_train= df_train[df_train['language']=="en"]
df_test= df_test[df_test['language']=="en"]


"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""##### Splitting data set in X and Y"""

def split_data(dataframe):
    y = dataframe['label']
    x = dataframe.drop(['label','Unnamed: 0','tweetId','tweetText','userId','imageId(s)','timestamp','username','link','language','filteredtweets','taggedData'],axis=1)
    return x,y

X_train, y_train = split_data(df_train)
X_test, y_test = split_data(df_test)


"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""##### Defining the metrics"""
score = pd.DataFrame(columns = ["Model_name",'accuracy', 'precision','recall','f1_score']) # Creating datafreame to store results of each model

def metric(ytrue,ypred,name):
    accuracy = accuracy_score(ytrue,ypred)
    precision = precision_score(ytrue,ypred)
    recall = recall_score(ytrue,ypred)
    f1 = f1_score(ytrue,ypred)

    d = {"Model_name":name,'accuracy': [accuracy], 'precision': [precision],'recall': [recall],'f1_score':[f1]}
    df = pd.DataFrame(data=d)
    return df

"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""## Predicting the labels using traditional ML algo"""


""" Logistic Regression"""

logr = linear_model.LogisticRegression()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear','newton-cholesky','sag']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# define search
search = GridSearchCV(logr, space, scoring='precision', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X_train, y_train)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

#Prediction on the training set
predicted_y_train = search.predict(X_train)
score = score.append(metric(y_train,predicted_y_train,"LogisticRegression_train")) # Updating the score dataframe

#Prediction on the test set
predicted_y_test = search.predict(X_test)
score = score.append(metric(y_test,predicted_y_test,"LogisticRegression_test")) # Updating the score dataframe


"""#### SVM Classifier"""

clf = SVC()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search spacefor different hyperparameters
space = dict()
space['C'] = [0.1,1,10,100,1000]
space['gamma'] = [1,0.1,0.5,0.001,0.0001]
space['kernel'] = ['rbf','linear', 'sigmoid']
# define search
search = GridSearchCV(clf, space, scoring='precision', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X_train, y_train)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
clf.fit(X_train, y_train)

#Prediction on the training set
predicted_y_train = search.predict(X_train)
score = score.append(metric(y_train,predicted_y_train,"SVMClassifier_train")) # Updating the score dataframe

#Prediction on the test set
predicted_y_test = search.predict(X_test)
score = score.append(metric(y_test,predicted_y_test,"SVMClassifier_test")) # Updating the score dataframe


"""#### Decision Tree"""

dtc = DecisionTreeClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['max_depth'] = [75,100,125,150,175,200]
space['min_samples_split'] = [10,11,15,20]
space['max_features'] = ['auto']
# define search
search = GridSearchCV(dtc, space, scoring='precision', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X_train, y_train)

#Prediction on the training set
predicted_y_train = search.predict(X_train)
score = score.append(metric(y_train,predicted_y_train,"DecisionTreeClassifier_train")) # Updating the score dataframe
#Prediction on the test set
predicted_y_test = search.predict(X_test)
score = score.append(metric(y_test,predicted_y_test,"DecisionTreeClassifier_test")) # Updating the score dataframe


"""#### Random Forest"""

rfc = RandomForestClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['max_depth'] = [125,130,135]
space['min_samples_split'] = [10,11,12]
space['max_features'] = ['auto']
# define search
search = GridSearchCV(rfc, space, scoring='precision', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X_train, y_train)
#Prediction on the training set
predicted_y_train = search.predict(X_train)
score = score.append(metric(y_train,predicted_y_train,"RandomForestClassifier_train")) # Updating the score dataframe
#Prediction on the test set
predicted_y_test = search.predict(X_test)
score = score.append(metric(y_test,predicted_y_test,"RandomForestClassifier_test")) # Updating the score dataframe


"""#### K Nearest Neighbour"""

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
#Prediction on the training set
predicted_y_train = knn.predict(X_train)
score = score.append(metric(y_train,predicted_y_train,"KNNClassifier_train")) # Updating the score dataframe
#Prediction on the test set 
predicted_y_test = knn.predict(X_test)
score = score.append(metric(y_test,predicted_y_test,"KNNClassifier_test")) # Updating the score dataframe

score.to_csv(path+"/method1.csv")