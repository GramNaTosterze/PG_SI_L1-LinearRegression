import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
#theta_best = [0, 0]
n = x_train.size
X = np.column_stack((x_train, np.ones(n)))
Y = y_train

X_T = X.transpose()

tmp1 = np.dot(X_T,X)
tmp2 = np.dot(X_T,Y)
theta_best = np.dot(np.linalg.inv(tmp1),tmp2)
theta_best[0], theta_best[1] = theta_best[1], theta_best[0]
# TODO: calculate error
def MSE(theta, X_t, Y_t):
	MSE = 0
	n = X_t.size
	for i in range(n):
		theta_x = float(theta[1])*X_t[i] + float(theta[0])
		MSE += (Y_t[i] - theta_x)**2

	MSE /= n
	return MSE
MSE_t = MSE(theta_best, x_train, y_train)
print(MSE_t)
	
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

X_o = X[:,0]
	
X_s = (X_o - np.mean(X_o))/np.std(X_o)
X_s = np.column_stack((X_s, np.ones(X_s.size)))
Y_s = (Y - np.mean(Y))/np.std(Y)

print("X_s: ", np.mean(X_s[:,0]), np.std(X_s[:,0]))
print("Y_s: ", np.mean(Y_s), np.std(Y_s))
# TODO: calculate theta using Batch Gradient Descent
def Grad(theta):
	tmp1 = 2*X_s.transpose()
	tmp2 = Y - np.dot(X_s, theta.transpose())
	return np.dot(tmp1,tmp2) / X_s.size

print(theta_best)
theta_best = np.array([0,0])
k = 0.001
for i in range(100):
	theta_best = theta_best - k*Grad(theta_best)
print(theta_best)

theta_best[0], theta_best[1] = theta_best[1], theta_best[0]
# TODO: calculate error
print(MSE(theta_best, x_train, y_train))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
