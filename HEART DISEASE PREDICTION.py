#!/usr/bin/env python
# coding: utf-8
#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
#Set to display all the columns in dataset
pd.set_option("display.max_columns", None)
#To run sql queries in DataFrame
import pandasql as psql
# In[2]:


#LOADING THE DATASET
data= pd.read_csv(r"C:\Users\Venkata Raghava\OneDrive\Documents\framingham.csv",header = 0)
data_bk=data.copy()
data.head(5)


# In[3]:


data.info()


# In[4]:


#checking the dataset
Target_count=data.TenYearCHD .value_counts()
print('Class 0:',Target_count[0])
print('Class 1:',Target_count[1])
print('proportion:',round(Target_count[0]/Target_count[1],2),':1')
print('Total Records',len(data))


# In[5]:


#Inference:As the ratio<10:1 the dataset is balanced dataset


# In[6]:


data.shape


# In[7]:


#identifing duplicate data
data_dup=data[data.duplicated(keep='last')]
data_dup


# In[8]:


#identifing the unique values
data.nunique()


# In[9]:


#Identifying the correlation among the variables
corr=data.corr()

#representing correlation using heatmap function of seabord class
sns.heatmap(corr)

Inference:
    Correlation between education and TenYearCHD is less than 0;hence we can drop education column
# In[10]:


# dropping unuseful columns
data=data.drop(['education'],axis=1)


# # DATA VISUALIZATION

# In[11]:


plt.figure(figsize=(12,9),dpi=85)
for i,col in enumerate(data.columns):
    plt.subplot(3,5,i+1)
    plt.title(col)
    plt.hist(data[col])
plt.show()


# In[12]:


#identifing missing data
data.isnull().sum()


# In[13]:


#DEALING WITH THE MISSING DATA
from sklearn.impute import KNNImputer
imputer=KNNImputer(missing_values=np.nan)
data['cigsPerDay']=imputer.fit_transform(data[['cigsPerDay']])
data['BPMeds']=imputer.fit_transform(data[['BPMeds']])
data['totChol']=imputer.fit_transform(data[['totChol']])
data['BMI']=imputer.fit_transform(data[['BMI']])
data['heartRate']=imputer.fit_transform(data[['heartRate']])
data['glucose']=imputer.fit_transform(data[['glucose']])
data.head(5)

data.isnull().sum()
# In[14]:


#converting the data from float to int datatype
data=data.astype(int)


# In[15]:


#identifying dependent and independent variables
IndepVar=[]
for col in data.columns:
    if col!='TenYearCHD':
        IndepVar.append(col)
TargetVar='TenYearCHD'
x=data[IndepVar]
y=data[TargetVar]



# In[16]:


#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=21)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[17]:


#scaling the values
from sklearn.preprocessing import MinMaxScaler
mmscaler=MinMaxScaler(feature_range=(0,1))
#x_train[cols] = mmscaler.fit_transform(x_train[cols])
x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)
#x_test[cols]= mmscaler.fit_transform(x_test[cols])
x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# # Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression
modelLR=LogisticRegression()
modelLR.fit(x_train,y_train)
y_pred=modelLR.predict(x_test)
y_pred_prob=modelLR.predict_proba(x_test)


# In[19]:


params=modelLR.get_params()
print(params)


# In[20]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#actual values
actual=y_test
#predicted values
predicted=y_pred

#confusion matrix
matrix=confusion_matrix(actual,predicted,labels=[1,0],sample_weight=None,normalize=None)
print("Confusion Matrix:\n",matrix)

#outcome values order in sklearn
tp,fn,fp,tn=confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print("Outcome values:\n",tp,fn,fp,tn)


# In[21]:


# Classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])
print('Classification report : \n', C_Report)

# Calculating the metrics
sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC liebetween -1 to +1.
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt
mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2),'%')
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, modelLR.predict_proba(x_test)
[:,1])
plt.figure()
#----------------------------------------------------
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[22]:


#adding the result to given dataset
Results = pd.DataFrame({'TenYearCHD_A':y_test, 'TenYearCHD_P':y_pred})

# Merge two Dataframes on index of both the dataframes
ResultsFinal = data.merge(Results, left_index=True, right_index=True)

# Display 5 records randomly
ResultsFinal.sample(5)


# # COMPARING WITH OTHER CLASSIFICATION ALGORITHMS

# In[23]:


#Loading the result file
em=pd.read_csv(r"C:\Users\Venkata Raghava\OneDrive\Documents\EMResults.csv",header= 0)
em.head()


# In[24]:


# Build the Calssification models and compare the results
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Create objects of classification algorithms with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)
ModelGNB = GaussianNB()
ModelSVM = SVC(probability=True)

# Evalution matrix for all the algorithms
MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelGNB,ModelSVM]
#MM = [ModelLR, ModelDC, ModelRF, ModelET]

for models in MM:
    
    # Train the model training dataset
    models.fit(x_train, y_train)
    
    # Prediction the model with test dataset
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    print('Model Name: ', models)
    # confusion matrix in sklearn
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values
    actual = y_test
    # predicted values
    predicted = y_pred

    # confusion matrix
    matrix = confusion_matrix(actual,predicted,
    labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn
    tp, fn, fp, tn =confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    C_Report = classification_report(actual,predicted,labels=[1,0])
    print('Classification report : \n', C_Report)

    # calculating the metrics
    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1.
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt
    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    print(mx)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :',round(specificity*100,2), '%' )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve
    from sklearn.metrics import roc_curve, roc_auc_score
    print('roc_auc_score:', round(roc_auc_score(actual, predicted),3))

    # ROC Curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    Model_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual,
    models.predict_proba(x_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label= 'Classification Model' % Model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------

    new_row = {'Model Name' : models,
               'True_Positive': tp,
               'False_Negative': fn,
               'False_Positive': fp,
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC':MCC,
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}

    em = em.append(new_row, ignore_index=True)
    #----------------------------------------------------------------------------------------------------------
    #======================================================================================================================>

how to analyze the best of all algorithms?
right side-->accuracy,f1 score,roc_auc_score(low)
Left side-->tp,tn,fp(low),fn(low)
# In[25]:


em.head(7)

Inference:
    We select RandomForest as better algorithm
    Scores -->TP=4,FP=2,ACCURACY=1,F1-SCORE=3,ROC_AUC=3
# In[26]:


y_pred1=ModelLR.predict(x_test)


# In[27]:


Results = pd.DataFrame({'TenYearCHD_A':y_test, 'TenYearCHD_P':y_pred1})

# Merge two Dataframes on index of both the dataframes
r = data.merge(Results, left_index=True, right_index=True)

# Display 5 records randomly
ResultsFinal.sample(5)


# In[30]:


#DATA VISUALIZATION OF TARGET VARIABLE
plt.figure(figsize=(8,8),dpi=75)
plt.subplot(2,1,1)
plt.title('Actual Data')
plt.pie(ResultsFinal['TenYearCHD_A'].value_counts(),labels=['Class 0','Class 1'])
plt.subplot(2,1,2)
plt.title('Predicted Data')
plt.pie(ResultsFinal['TenYearCHD_P'].value_counts(),labels=['Class 0','Class 1'])
plt.show()

