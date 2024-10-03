# EX1 Implementation of Univariate Linear Regression to fit a straight line using Least Squares
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: K.Nishal
RegisterNumber:2305001021  
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1 ml.csv')
df.head()
plt.scatter(df['X'],df['Y'])
plt.xlabel('x')
plt.ylabel('y')
X=df['X']
Y=df['Y']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['X']],df['Y'],test_size=0.2)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

X=df.iloc[:,0].values
Y=df.iloc[:,-1].values
X

plt.scatter(df['X'],df['Y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X,model.predict(df[['X']]),color='red')

m=model.coef_
m

b=model.intercept_
b
```

## Output:
![image](https://github.com/user-attachments/assets/7bc221b2-a8ef-4f88-ad20-c21293b097ae)
![image](https://github.com/user-attachments/assets/4c1837ae-936f-429f-9638-9741f71341bb)
![image](https://github.com/user-attachments/assets/9a5206c1-9a57-4a3a-93b4-d4284d2d72ca)
![image](https://github.com/user-attachments/assets/376ffbc0-8719-4a3d-bf67-bcc3fd697588)
![image](https://github.com/user-attachments/assets/2a6f2c5e-7f07-4693-9d9e-1b3ab97cc85a)
![image](https://github.com/user-attachments/assets/49b71425-e4d9-4098-8d28-4eddfd2ac319)
![image](https://github.com/user-attachments/assets/d406a152-0c2e-4c4a-a589-0b524d1b256b)









## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
