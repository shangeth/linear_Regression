import pandas as pd
from sklearn.cross_validation import train_test_split
from Linear_Regression_Gradient_Descent import gradient_descent_linear_regression
import numpy as np
from sklearn.metrics import accuracy_score



def run():
    df = pd.read_csv("data.csv")
    y = df.calories
    X = df.drop('calories',axis=1)

    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.1)
    clf = gradient_descent_linear_regression()

    clf.fit(X_tr,y_tr)
    # pred = clf.predict(X_te)
    # print("predictions for X-test datas " ,pred)
    r = clf.r_squared()
    print("r_square value = ",float(r))
    clf.plot(X_te,y_te)



if __name__=="__main__":
    run()
