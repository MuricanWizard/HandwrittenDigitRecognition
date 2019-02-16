# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:20:26 2018

@author: Akhil
"""

from mnist import MNIST
import random
from PIL import Image
import numpy as np
from sklearn import tree

mndata = MNIST('DigitRecognition')

images, labels = mndata.load_training()

index = random.randrange(0, len(images))
'''
print(mndata.display(images[index]))
print("This is "+ str(labels[index]))
'''
'''
for i in range(len(images)):
    temp = images[i]
    temp=np.reshape(temp,(28,28))
    temp=temp*255
    im = Image.fromarray(temp).convert('L')
    im.save("DigitRecognition/Testing/img" + str(i) + ".png")

for i in range(len(images)):
    imf = Image.open("DigitRecognition/Training/img" + str(i) + ".png")
    im5 = imf.resize((15, 15), 1)
    im5.save("DigitRecognition/Training/img" + str(i) + ".png")
'''
def arrayClassifierTesting(ran):
    testing = []
    for ind in range(ran):
        img = Image.open( "DigitRecognition/Testing/img" + str(ind) + ".png" )
        img.load()
        data = np.asarray( img, dtype="int32" )
        img.close()
        #data = np.reshape(data, (225, 1))
        newData = []
        for i in data:
            newData.extend(i)
        testing.append(newData)
    return testing
        
def arrayClassifierTraining(ran):
    training = []
    for ind in range(ran):
        img = Image.open( "DigitRecognition/Training/img" + str(ind) + ".png" )
        img.load()
        data = np.asarray( img, dtype="int32" )
        img.close()
        #data = np.reshape(data, (225, 1))
        newData = []
        for i in data:
            newData.extend(i)
        training.append(newData)
    return training
weights1 = []
bias1 = []
'''
def NNLayer1():
    for i in range(225):
        tempw = []
        for j in range(8):
            tempw.append(np.random.randn())
        weights1.append(tempw)
    for i in range(8):
        bias1.append(np.random.randn())
    del tempw    
def activation(neurons1, weights1, bias1):
    secondLayer = []
    temp = 0
    for i in range(8):
        temp = 0
        for j in range(len(neurons1[0])):
            temp += sigmoid(neurons1[0][j])*weights1[j][i]
        secondLayer.append(sigmoid(temp + bias1[i]))
    return secondLayer
'''
'''
def NNLayer1(neurons1):
    (neurons1, weights1, bias1) = ([], [], [])
    for i in range(225):
        tempw = []
        for j in range(8):
            tempw.append(np.random.randn())
        weights1.append(tempw)
        for i in range(8):
            bias1.append(np.random.rand())
    activationLayer1(neurons1, weights1, bias1)
            
            
def activationLayer1(neurons1, weights1, bias1):
     secondLayer = []
     temp = 0
     for i in range(8):
         temp = 0
         for j in range(neurons1[0]):
             temp += neurons1[0][j]*weights1[j][i]
         secondLayer.append(sigmoid(temp + bias1[i]))
         return secondLayer
'''
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def arrayClassifierPrediction():
    prediction = []
    img2 = Image.open("DigitRecognition/Predict/img0.png")
    img2.load()
    prdata = np.asarray(img2, dtype="int32")
    img2.close()
    newprdata = []
    for i in prdata:
        newprdata.extend(i)
    prediction.append(newprdata)
    return prediction

training = arrayClassifierTraining(60000)
    
predictor = tree.DecisionTreeClassifier()
label = []
for i in labels:
    label.append([i])
#testing = arrayClassifierTesting()    
predictor.fit(training, label)
prediction = arrayClassifierPrediction()
print(predictor.predict(prediction))
#print(len(training))
#print(len(testing))
#print (NNLayer1(training))
#NNLayer1()
#print(activation(training, weights1, bias1))
