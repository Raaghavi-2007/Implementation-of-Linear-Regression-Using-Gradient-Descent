# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset from the CSV file, and initialize the input feature and output variable in separate arrays.
3. Scale the features using a standard scaler to normalize the data. 
4. Initialize the weight and bias, define the learning rate and number of iterations, and set up the Mean Squared Error loss function.
5. Predict the model to provide output values, compute the loss, calculate gradients, and update the parameters using gradient descent.
6. Plot the loss curve and regression line with the data points, and display the final weight and bias values.

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

for _ in range(epochs):
    y_hat = w * x + b

    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Actual Data")
plt.plot(x, y_hat, color="green", label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit (Scaled)")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
<img width="1301" height="583" alt="image" src="https://github.com/user-attachments/assets/deb58b77-6a12-4e0e-8e72-f38b625ee3a7" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
