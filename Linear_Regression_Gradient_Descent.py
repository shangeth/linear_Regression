import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

class gradient_descent_linear_regression:

    def __init__(self,learning_rate=0.0001,initial_b=0,initial_m=0,num_iterations=10000):
        self.learning_rate = learning_rate
        self.initial_b = initial_b
        self.initial_m = initial_m
        self.num_iterations = num_iterations

    def compute_error(self,b, m, points):
        totalError = 0
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            totalError += (y - (m * x + b)) ** 2
        return totalError / float(len(points))

    def step_gradient(self,b_current, m_current, X_train,y_train, learning_rate):
        b_gradient = 0
        m_gradient = 0
        N = float(len(X_train))

        for i in range(0, len(X_train)):
            [x] = np.array(X_train)[i]
            y = np.array(y_train)[i]
            b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
            m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

        new_b = b_current - (learning_rate * b_gradient)
        new_m = m_current - (learning_rate * m_gradient)
        return [new_b, new_m]

    def gradient_descent_runner(self,X_train,y_train, starting_b, starting_m, learning_rate, num_iterations):
        b = starting_b
        m = starting_m

        for i in range(num_iterations):
            b, m = self.step_gradient(b, m, X_train,y_train, learning_rate)
        return [b, m]

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        [b,m] = self.gradient_descent_runner(self.X_train,self.y_train,self.initial_b,self.initial_m,self.learning_rate,self.num_iterations)
        self.b = b
        self.m = m


    def squared_error(self,ys_orig, ys_line):
        sum = 0
        for i in range(len(ys_line)):
            sum += (ys_line[i]-ys_orig[i])**2

        return sum

    def coefficient_of_determination(self,ys_orig, ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        squared_error_regr = self.squared_error(ys_orig, ys_line)
        squared_error_y_mean = self.squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr / squared_error_y_mean)

    def r_squared(self):

        return self.coefficient_of_determination(np.array(self.y_train), self.m * np.array(self.X_train) + self.b )

    def plot(self,X_test,y_test):
        regression_line = self.m * np.array(self.X_train) + self.b
        plt.scatter(np.array(self.X_train), np.array(self.y_train))
        # plt.scatter(p, predict_y, color="black")
        plt.plot(np.array(self.X_train), regression_line)
        plt.scatter(np.array(X_test),np.array(y_test),c="red",s=200)
        plt.scatter(np.array(X_test),self.m* np.array(X_test) + self.b,c="black",s=200)
        plt.show()