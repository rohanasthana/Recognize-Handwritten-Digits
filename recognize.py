import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import random

train=pd.read_csv("train.csv").as_matrix()
print(len(train))

xtrain=train[0:35000,1:]
ytrain=train[0:35000,0]
xtest=train[35000:,1:]
ytest=train[35000:,0]
x=0
clf=MLPClassifier()
clf.fit(xtrain,ytrain)
predict=clf.predict(xtest)
for i in range(0,7000):
    if predict[i]==ytest[i]:
        x+=1
print("Accuracy is ",x/7000*100," %")

test=pd.read_csv("test.csv").as_matrix()
predict_test=clf.predict(test)
while True:
    i=random.randint(0,28000)
    print("Predicted Value: ",predict_test[i])
    d=test[i]
    d.shape=(28,28)
    plt.imshow(d,cmap='gray')
    plt.show()
    
        

