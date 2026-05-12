# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAMASRI K
RegisterNumber:  212224040267
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

### TOP 5 ELEMENTS
<img width="1217" height="238" alt="image" src="https://github.com/user-attachments/assets/d43e2329-c78b-4ff9-899d-85f2160e6cd9" />

<img width="1115" height="237" alt="image" src="https://github.com/user-attachments/assets/608d089f-8274-4075-8ce7-a57f5d673a42" />

#### PRINT DATA
<img width="972" height="457" alt="image" src="https://github.com/user-attachments/assets/37d3a492-d41a-4db2-95c3-3a15b14e107c" />


#### DATA_STATUS
<img width="896" height="460" alt="image" src="https://github.com/user-attachments/assets/614da568-ace2-4b7f-9b1b-e74138a3cf13" />

#### Y_PREDICTION ARRAY
<img width="670" height="273" alt="image" src="https://github.com/user-attachments/assets/2f93f881-d621-4e34-a84a-411b40ec24a7" />

#### CONFUSION ARRAY
<img width="843" height="56" alt="image" src="https://github.com/user-attachments/assets/38016d54-442f-41c4-b7b4-1fc555ad59e4" />

#### ACCURACY VALUE
<img width="365" height="44" alt="image" src="https://github.com/user-attachments/assets/e0619f6a-2126-43ad-8bbb-22b6aca62d0a" />
<img width="608" height="59" alt="image" src="https://github.com/user-attachments/assets/4ade8b39-13c2-426e-91d6-e603b051d860" />

#### CLASSFICATION REPORT
<img width="675" height="187" alt="image" src="https://github.com/user-attachments/assets/3eb0abe2-4fb9-4d86-ab4e-8f677ab5f3e4" />

#### PREDICTION
<img width="395" height="49" alt="image" src="https://github.com/user-attachments/assets/69d106d9-b346-4422-beaa-6ccebef8f642" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
