'''
Show limitations of logistic function
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

pwd = os.getcwd()
plt.rcParams.update({'font.size': 18})

def model(x):
    return 1 / (1 + np.exp(-x))

# Show good case
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x  = np.reshape(x, (len(x), 1))
y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
y = np.reshape(y, (len(y), 1))

lr = LogisticRegression()
lr.fit(x, y)

plt.figure(figsize=(10, 5))
plt.scatter(x.ravel(), y, color='k', label='Data')

x_test = np.linspace(-5, 15, 300)
loss = model(x_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(x_test, loss, color='r', linewidth=2, label='Logistic Sigmoid')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim(0, 13)
plt.grid()
plt.legend()
plt.title('Logistic Regression Sigmoid with Linear Data')
save_fig = os.path.join(pwd, 'figs', 'model_limitations')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
plt.savefig(os.path.join(save_fig, 'sigmoid_linear.png'), dpi=150)

# Show bad case
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x  = np.reshape(x, (len(x), 1))
y = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
y = np.reshape(y, (len(y), 1))

lr = LogisticRegression()
lr.fit(x, y)

plt.figure(figsize=(10, 5))
plt.scatter(x.ravel(), y, color='k', label='Data')

x_test = np.linspace(-5, 15, 300)
loss = model(x_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(x_test, loss, color='r', linewidth=2, label='Logistic Sigmoid')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim(0, 13)
plt.grid()
plt.legend(loc='center left')
plt.title('Logistic Regression Sigmoid with Nonlinear Data')
plt.savefig(os.path.join(save_fig, 'sigmoid_nonlinear.png'), dpi=150)


