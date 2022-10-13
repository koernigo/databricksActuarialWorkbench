# Databricks notebook source
# MAGIC %md
# MAGIC # **Classification With Model Interpretation**  💯 💯

# COMMAND ----------

# MAGIC %md
# MAGIC # **Importing Modules**

# COMMAND ----------

from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,r2_score
import warnings
from mlxtend.classifier import StackingClassifier
import missingno as msno
from sklearn.ensemble import VotingClassifier
import shap
shap.initjs()
import lime
from lime import lime_tabular
warnings.simplefilter('ignore')
import os
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# COMMAND ----------

# MAGIC %md
# MAGIC # **Reading Data**

# COMMAND ----------

insurance_df = pd.read_csv('../input/prudential-life-insurance-assessment/train.csv.zip', index_col='Id')
insurance_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Shape**

# COMMAND ----------

insurance_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # **Distribution of Target Variable**

# COMMAND ----------

insurance_df['Response'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC > **Class imbalance can be seen here. Also there 8 categories, lets combine them to 3 categories**

# COMMAND ----------

sns.countplot(x=insurance_df['Response']);

# COMMAND ----------

# MAGIC %md
# MAGIC **Response 8 has highest values and 3 has the least**

# COMMAND ----------

# MAGIC %md
# MAGIC # **Processing Target Variable**

# COMMAND ----------

#Combining the Categores to 3 categories
insurance_df['Modified_Response']  = insurance_df['Response'].apply(lambda x : 0 if x<=7 and x>=0 else (1 if x==8 else -1))

# COMMAND ----------

sns.countplot(x= insurance_df['Modified_Response']);

# COMMAND ----------

# MAGIC %md
# MAGIC > **Still some imbalance can be seen**

# COMMAND ----------

# MAGIC %md
# MAGIC # **Removing old target variable**

# COMMAND ----------

# Dropping old response columns
insurance_df.drop('Response',axis = 1, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Making categorical and numerical columns list**

# COMMAND ----------

# Making lists with categorical and numerical features.
categorical =  [col for col in insurance_df.columns if insurance_df[col].dtype =='object']

numerical = categorical =  [col for col in insurance_df.columns if insurance_df[col].dtype !='object']

# COMMAND ----------

# MAGIC %md
# MAGIC # **Visualizations On Categorical Features**

# COMMAND ----------

# Doing count plots for categorical
for col in categorical:
    counts = insurance_df[col].value_counts().sort_index()
    if len(counts) > 10 and len(counts) < 50 :
      fig = plt.figure(figsize=(30, 10))
    elif len(counts) >50 :
      continue
    else:
      fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC > **D3 has the highest frequencies**
# MAGIC 
# MAGIC > Most of the features here are unbalanced.

# COMMAND ----------

fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.distplot(insurance_df['Employment_Info_1'], ax=axes[0])
sns.boxplot(insurance_df['Employment_Info_1'], ax=axes[1])

# COMMAND ----------

# MAGIC %md
# MAGIC > Right skewed.
# MAGIC 
# MAGIC > Outliers can be seen.

# COMMAND ----------

fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.distplot(insurance_df['Employment_Info_4'], ax=axes[0])
sns.boxplot(insurance_df['Employment_Info_4'], ax=axes[1])

# COMMAND ----------

fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.distplot(insurance_df['Employment_Info_6'], ax=axes[0])
sns.boxplot(insurance_df['Employment_Info_6'], ax=axes[1])

# COMMAND ----------

fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.distplot(insurance_df['Family_Hist_4'], ax=axes[0])
sns.boxplot(insurance_df['Family_Hist_4'], ax=axes[1])

# COMMAND ----------

# MAGIC %md
# MAGIC # **Checking Correlation For Features greater than .8**

# COMMAND ----------

# I just checked correlated feature with greater than .8 here 
corr = insurance_df.corr()
corr_greater_than_80 = corr[corr>=.8]
corr_greater_than_80


# COMMAND ----------

plt.figure(figsize=(12,8))
sns.heatmap(corr_greater_than_80, cmap="Reds");

# COMMAND ----------

# MAGIC %md
# MAGIC ### **CONCLUSION**
# MAGIC 
# MAGIC > BMI and Weight are highly correlated, which makes sense also as these 2 features are directly proprtional.
# MAGIC 
# MAGIC > Ins_Age and Family_Hist_4, Family_Hist_2 highly correlated
# MAGIC 
# MAGIC > Although, I am not going to perform any transformation on any feature or drop any as these are tree based models and they don't get affected by correlation much because of their non parametric nature.

# COMMAND ----------

#setting max columns to 200
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Null Value Check**

# COMMAND ----------

#checking percentage of missing values in a column
missing_val_count_by_column = insurance_df.isnull().sum()/len(insurance_df)

print(missing_val_count_by_column[missing_val_count_by_column > 0.4].sort_values(ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC # **Removing unimportant column**

# COMMAND ----------

# Dropping all columns in which greater than 40 percent null values
insurance_df = insurance_df.dropna(thresh=insurance_df.shape[0]*0.4,how='all',axis=1)

# COMMAND ----------

# Does not contain important information
insurance_df.drop('Product_Info_2',axis=1,inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # **X and Y split**

# COMMAND ----------

# Data for all the independent variables
X = insurance_df.drop(labels='Modified_Response',axis=1)

# Data for the dependent variable
Y = insurance_df['Modified_Response']

# COMMAND ----------

# MAGIC %md
# MAGIC # **Filling Remaining Missing Values**

# COMMAND ----------

# Filling remaining missing values with mean
X = X.fillna(X.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC # **Train Test Split**

# COMMAND ----------

# Train-test split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Shapes of Train and Test Data**

# COMMAND ----------

# Check the shape of train dataset
print(X_train.shape,Y_train.shape)

# Check the shape of test dataset
print(X_test.shape, Y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Some Important functions that I will be using throughout**

# COMMAND ----------

# Utility Functions
def check_scores(model, X_train, X_test ):
  # Making predictions on train and test data

  train_class_preds = model.predict(X_train)
  test_class_preds = model.predict(X_test)


  # Get the probabilities on train and test
  train_preds = model.predict_proba(X_train)[:,1]
  test_preds = model.predict_proba(X_test)[:,1]


  # Calculating accuracy on train and test
  train_accuracy = accuracy_score(Y_train,train_class_preds)
  test_accuracy = accuracy_score(Y_test,test_class_preds)

  print("The accuracy on train dataset is", train_accuracy)
  print("The accuracy on test dataset is", test_accuracy)
  print()
  # Get the confusion matrices for train and test
  train_cm = confusion_matrix(Y_train,train_class_preds)
  test_cm = confusion_matrix(Y_test,test_class_preds )

  print('Train confusion matrix:')
  print( train_cm)
  print()
  print('Test confusion matrix:')
  print(test_cm)
  print()

  # Get the roc_auc score for train and test dataset
  train_auc = roc_auc_score(Y_train,train_preds)
  test_auc = roc_auc_score(Y_test,test_preds)

  print('ROC on train data:', train_auc)
  print('ROC on test data:', test_auc)
  
  # Fscore, precision and recall on test data
  f1 = f1_score(Y_test, test_class_preds)
  precision = precision_score(Y_test, test_class_preds)
  recall = recall_score(Y_test, test_class_preds) 
  
  
  #R2 score on train and test data
  train_log = log_loss(Y_train,train_preds)
  test_log = log_loss(Y_test, test_preds)

  print()
  print('Train log loss:', train_log)
  print('Test log loss:', test_log)
  print()
  print("F score is:",f1 )
  print("Precision is:",precision)
  print("Recall is:", recall)
  return model, train_auc, test_auc, train_accuracy, test_accuracy,f1, precision,recall, train_log, test_log


def check_importance(model, X_train):
  #Checking importance of features
  importances = model.feature_importances_
  
  #List of columns and their importances
  importance_dict = {'Feature' : list(X_train.columns),
                    'Feature Importance' : importances}
  #Creating a dataframe
  importance_df = pd.DataFrame(importance_dict)
  
  #Rounding it off to 2 digits as we might get exponential numbers
  importance_df['Feature Importance'] = round(importance_df['Feature Importance'],2)
  return importance_df.sort_values(by=['Feature Importance'],ascending=False)

def grid_search(model, parameters, X_train, Y_train):
  #Doing a grid
  grid = GridSearchCV(estimator=model,
                       param_grid = parameters,
                       cv = 2, verbose=2, scoring='roc_auc')
  #Fitting the grid 
  grid.fit(X_train,Y_train)
  print()
  print()
  # Best model found using grid search
  optimal_model = grid.best_estimator_
  print('Best parameters are: ')
  pprint( grid.best_params_)

  return optimal_model



# This function will show how a feature is pushing towards 0 or 1
def interpret_with_lime(model, X_test):
  # New data
  interpretor = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='classification')
  

  exp = interpretor.explain_instance(
      data_row=X_test.iloc[10], 
      predict_fn=model.predict_proba
  )

  exp.show_in_notebook(show_table=True)

# This gives feature importance
def plot_feature_importance(model, X_train):
  # PLotting features vs their importance factors
  fig = plt.figure(figsize = (15, 8))
  
  # Extracting importance values
  values =check_importance(model, X_train)[check_importance(model, X_train)['Feature Importance']>0]['Feature Importance'].values
  
  
  # Extracting importance features
  features = check_importance(model, X_train)[check_importance(model, X_train)['Feature Importance']>0]['Feature'].values

  plt.bar(features, values, color ='blue',
          width = 0.4)
  plt.xticks( rotation='vertical')
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Random Forest**

# COMMAND ----------


# Number of trees
n_estimators = [50,80,100]

# Maximum depth of trees
max_depth = [4,6,8]

# Minimum number of samples required to split a node
min_samples_split = [50,100,150]

# Minimum number of samples required at each leaf node
min_samples_leaf = [40,50]

# Hyperparameter Grid
rf_parameters = {'n_estimators' : n_estimators,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf}

pprint(rf_parameters)

#finding the best model
rf_optimal_model = grid_search(RandomForestClassifier(), rf_parameters, X_train, Y_train)

# COMMAND ----------

# Getting scores from all the metrices
rf_model, rf_train_auc, rf_test_auc, rf_train_accuracy, rf_test_accuracy,rf_f1, rf_precision,rf_recall,rf_train_log, rf_test_log = check_scores(rf_optimal_model, X_train, X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC # **Feature Importance For Random Forest**

# COMMAND ----------

# Getting the feature importance for all the features
check_importance(rf_model, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Plotting only those features which are contributing something**

# COMMAND ----------

# PLotting only those features which are contributing something
plot_feature_importance(rf_model, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **CONCLUSION:**
# MAGIC 
# MAGIC > BMI, weight, Medical_History_23, Medical_History_4 and Medical_Keyword_15 seems to be important features according to random forest.
# MAGIC 
# MAGIC > Also, only these features are contributing to the model prediction. Some features can be elmininated which are not contributing on further investigation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Model Interpretability For Random Forest**

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Using Lime**

# COMMAND ----------


# Interpretting the model using lime
interpret_with_lime(rf_model,X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Using Shap**

# COMMAND ----------


# Interpretting the model using shaply
X_shap=X_train

rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_shap)
shap.summary_plot(rf_shap_values, X_shap, plot_type="bar")

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Findings**
# MAGIC 
# MAGIC > Medical keyword 15,medical history 9,  Wt, medical history 3 all pushing towards 1.
# MAGIC 
# MAGIC > Orange ones are pusing towards 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Dependence Plots**

# COMMAND ----------

# Plotting for top 5 features
top_vars = ['BMI','Medical_Keyword_15','Medical_History_4','Wt','Medical_History_23']
index_top_vars =[list(X_train.columns).index(var) for var in top_vars]

for elem in index_top_vars:
    shap.dependence_plot(elem, rf_shap_values[0], X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Findings**
# MAGIC 
# MAGIC > With high medical history 23 and low bmi we get class 1

# COMMAND ----------

# MAGIC %md
# MAGIC # **Gradient Boosting**

# COMMAND ----------

#finding the best model
gb_parameters ={
    "n_estimators":[5,50,250],
    "max_depth":[1,3,5,7],
    "learning_rate":[0.01,0.1,1]
}

pprint(gb_parameters)

gb_optimal_model = grid_search(GradientBoostingClassifier(), gb_parameters, X_train, Y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Feature Importance For Gradient Boosting**

# COMMAND ----------

# Getting the scpres for all the score metrics used here
gb_model, gb_train_auc, gb_test_auc, gb_train_accuracy, gb_test_accuracy,gb_f1, gb_precision,gb_recall,gb_train_log, gb_test_log = check_scores(gb_optimal_model, X_train, X_test )

# COMMAND ----------

# Getting feature importance
check_importance(gb_model, X_train)

# COMMAND ----------

# PLotting only those features which are contributing something
plot_feature_importance(gb_model, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **CONCLUSION:**
# MAGIC 
# MAGIC > BMI, weight, Medical_History_23, Medical_History_4 and Medical_Keyword_15 seems to be the most important 5 features according to Gradient boosting.

# COMMAND ----------

# MAGIC %md
# MAGIC # **Model Interpretability For Gradient Boosting**

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Using Lime**

# COMMAND ----------

# Interpretting the model using lime
interpret_with_lime(gb_model,X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Using Shap**

# COMMAND ----------


# Interpretting the model using shaply
X_shap=X_train

gb_explainer = shap.TreeExplainer(gb_model)
gb_shap_values = gb_explainer.shap_values(X_shap)
shap.summary_plot(gb_shap_values, X_shap, plot_type="dot")

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Findings**
# MAGIC > BMI is pushing models prediction towards 0.
# MAGIC 
# MAGIC >Medical keyword 15 is pushing towards 1. However, medical keyword 4 is pushing towards 0.
# MAGIC 
# MAGIC > Also, according to feature plot Wt. was in top 5 most important features, same isn't followed here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Dependence Plots**

# COMMAND ----------

#PLotting for top 5 features
top_vars = ['BMI','Medical_Keyword_15','Medical_History_4','Product_Info_4','Medical_History_23']
index_top_vars =[list(X_train.columns).index(var) for var in top_vars]

for elem in index_top_vars:
    shap.dependence_plot(elem, gb_shap_values, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Findings**
# MAGIC 
# MAGIC > For low BMI and high medical history 23 we get class as 1.

# COMMAND ----------

# MAGIC %md
# MAGIC # **XGBOOST**

# COMMAND ----------

# Parameter grid for xgboost
xgb_parameters = {'max_depth': [1,3,5], 'n_estimators': [2,5,10], 'learning_rate': [.01 , .1, .5]}
print('XGB parameters areL:')
pprint(xgb_parameters)
#finding the best model
xgb_optimal_model = grid_search(XGBClassifier(), xgb_parameters, X_train, Y_train)




# COMMAND ----------

# Getting the scores for all the score metrics used here
xgb_model, xgb_train_auc, xgb_test_auc, xgb_train_accuracy, xgb_test_accuracy,xgb_f1, xgb_precision,xgb_recall,xgb_train_log, xgb_test_log= check_scores(xgb_optimal_model, X_train, X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC # **Feature Importance For XGBoost**

# COMMAND ----------

# Getting feature importance

check_importance(xgb_model, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Conclusion:**
# MAGIC 
# MAGIC > Same trend is seen here.
# MAGIC 
# MAGIC > They all are giving similar scores also so it could be that same features are contributing the most thus similar scores.

# COMMAND ----------

# MAGIC %md
# MAGIC # **Model Interpretability for XGBoost**

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Using Shap**

# COMMAND ----------


# Interpretting the model using shaply

xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_shap)
shap.summary_plot(xgb_shap_values, X_shap, plot_type="dot")

# COMMAND ----------

# MAGIC %md
# MAGIC > Again BMI is pushing towards class 0.
# MAGIC 
# MAGIC > MEdical history 4 pushing towards class 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Dependence Plots**

# COMMAND ----------

#PLotting for top 5 features
top_vars = ['BMI','Medical_Keyword_15','Medical_History_4','Product_Info_4','Medical_History_23']
index_top_vars =[list(X_train.columns).index(var) for var in top_vars]

for elem in index_top_vars:
    shap.dependence_plot(elem, xgb_shap_values, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC > For product info 4 and wt we see some interesting trend

# COMMAND ----------

# MAGIC %md
# MAGIC # **Logistic Regression**

# COMMAND ----------


# Parameter grid for Logistic Regression
solvers = ['lbfgs']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
lr_parameters = dict(solver=solvers,penalty=penalty,C=c_values)# define grid search

#finding the best model
lr_optimal_model = grid_search(LogisticRegression( max_iter=5000), lr_parameters, X_train, Y_train)



# COMMAND ----------

# Getting the scores for all the score metrics used here

lr_model, lr_train_auc, lr_test_auc, lr_train_accuracy, lr_test_accuracy,lr_f1, lr_precision, lr_recall,lr_train_log, lr_test_log = check_scores(lr_optimal_model, X_train, X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC # **Feature Importance For Logistic Regression**

# COMMAND ----------

# Making a dataframe with coefficients and the feature names respectively
importance_df_lr = pd.concat([ pd.DataFrame(data =((X_train.columns).values).reshape(-1,1), columns = ['Feature']), pd.DataFrame(data =np.round(lr_optimal_model.coef_,2).reshape(-1,1), columns = ['Feature Importance'])], axis=1 )
importance_df_lr.sort_values(by=['Feature Importance'],ascending=False, inplace = True)
importance_df_lr

# COMMAND ----------

# Plotting feature vs importance
fig = plt.figure(figsize = (15, 8))

values =importance_df_lr[importance_df_lr['Feature Importance']>0]['Feature Importance'].values

features = importance_df_lr[importance_df_lr['Feature Importance']>0]['Feature'].values

plt.bar(features, values, color ='blue',
          width = 0.4)
plt.xticks( rotation='vertical')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Conclusion**
# MAGIC 
# MAGIC > **And again the same pattern when doing feature importance**

# COMMAND ----------

# MAGIC %md
# MAGIC # **Model Interpretability for logistic regression**

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Using Lime**

# COMMAND ----------


# Interpretting the model using lime
interpret_with_lime(lr_model,X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Findings**
# MAGIC 
# MAGIC > Only BMI and medical history 4 pushing towards class 0

# COMMAND ----------

# MAGIC %md
# MAGIC # **Max Voting Model**

# COMMAND ----------

# Appending all the models to estimators list
estimators = []

estimators.append(('logistic', lr_optimal_model))
estimators.append(('XGB', xgb_optimal_model))
estimators.append(('GB', gb_optimal_model))
estimators.append(('rf', rf_optimal_model))

# create the voting model
voting_model = VotingClassifier(estimators, voting='soft')

voting_model.fit(X_train, Y_train)


# COMMAND ----------

# Getting all the scores and errors
voting_model, voting_train_auc, voting_test_auc, voting_train_accuracy, voting_test_accuracy, voting_f1, voting_precision, voting_recall, voting_train_log, voting_test_log = check_scores(voting_model, X_train, X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC # **Stacked Model**

# COMMAND ----------

#Building a stacked classifier
stacked_classifier = StackingClassifier(classifiers =[lr_optimal_model, xgb_optimal_model, gb_model], meta_classifier = RandomForestClassifier(), use_probas = True, use_features_in_secondary = True)

# training of stacked model
stacked_model = stacked_classifier.fit(X_train, Y_train)   


# COMMAND ----------

stacked_model, stacked_train_auc, stacked_test_auc, stacked_train_accuracy, stacked_test_accuracy, stacked_f1, stacked_precision, stacked_recall, stacked_train_log, stacked_test_log = check_scores(stacked_model, X_train, X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC # **Models And Their Accuracies**

# COMMAND ----------

# Making a dataframe of all the scores for every model

scores_ = [("Random Forest", rf_train_auc, rf_test_auc, rf_train_accuracy, rf_test_accuracy,rf_train_log, rf_test_log,rf_f1, rf_precision, rf_recall),
("Gradient Boosting",  gb_train_auc, gb_test_auc, gb_train_accuracy, gb_test_accuracy,gb_train_log, gb_test_log,gb_f1, gb_precision,gb_recall,),
("XG Boost", xgb_train_auc, xgb_test_auc, xgb_train_accuracy, xgb_test_accuracy,xgb_train_log, xgb_test_log,xgb_f1, xgb_precision, xgb_recall),
("Logistic Regression", lr_train_auc, lr_test_auc, lr_train_accuracy, lr_test_accuracy,lr_train_log, lr_test_log,lr_f1, lr_precision, lr_recall,),
("Voting Classifier", voting_train_auc, voting_test_auc, voting_train_accuracy, voting_test_accuracy, voting_train_log, voting_test_log, voting_f1, voting_precision, voting_recall),
("Stacked Model", stacked_train_auc, stacked_test_auc, stacked_train_accuracy, stacked_test_accuracy, stacked_train_log, stacked_test_log, stacked_f1, stacked_precision, stacked_recall)]

Scores_ =pd.DataFrame(data = scores_, columns=['Model Name', 'Train ROC', 'Test ROC', 'Train Accuracy', 'Test Accuracy', 'Train Log Loss','Test Log Loss','F-Score', 'Precision','Recall',])
Scores_.set_index('Model Name', inplace = True)

Scores_


# COMMAND ----------

# MAGIC %md
# MAGIC # **Final Results**
# MAGIC 
# MAGIC > **Gradient Boosting, Voting Classifier and Stacked models are performing really well. Their train and test errors and also the roc scores and f scores are really close and good.**
