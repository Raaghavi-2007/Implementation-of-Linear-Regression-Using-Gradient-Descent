# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Raaghavi S
RegisterNumber: 25012715  
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex3.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

x = (x - np.mean(x)) / np.std(x)

w = 0.0          
b = 0.0         
alpha = 0.01   
epochs = 100
n = len(x)

losses = []

for i in range(epochs):
    y_hat = w * x + b

    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w-=  alpha * dw
    b-=  alpha * db

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color='blue')
plt.xlabel("R&D Spend")
plt.ylabel("Profit (MSE)")
plt.title("R&D Spend vs Profit")

plt.subplot(1, 2, 2)
plt.scatter(x, y,color="red",label="Actual Data")
plt.plot(x, y_hat, label="Regression Line",color='green')
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit (Scaled)")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.show()


print("Final Weight (w):", w)
print("Final Bias (b):", b)
```

## Output:

<img width="1316" height="649" alt="Screenshot 2026-01-30 113343" src="https://github.com/user-attachments/assets/0b65df79-40fa-4faa-b20f-2343192da5ca" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
