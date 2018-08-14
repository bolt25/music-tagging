import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#add = ("C:\\Users\dharm\Desktop\dataset.csv")

#names=['filename', 'tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff',
#       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
#       'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label']

dataset= np.genfromtxt("C:\\Users\dharm\Desktop\dataset.csv", delimiter=",")

array=dataset[1:,]

x=array[0:,1:-1]

y_raw=array[0: , -1]
y=y_raw.reshape((199,1))

#y= np.zeros((199,1))

#print(y.shape)

"""for i in range(199):
    y[i][0]= y_raw[i]"""
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2,random_state=0)

input_size= x_train.shape[1]



weights= np.random.rand(input_size, 1)

print(weights.shape,input_size)

output=np.dot(x_train,weights)

def sigmoid(output):
    return 1/(1+np.exp(-output))

h=sigmoid(output)

m=len(y)

alpha=0.01


def cost(h,y):
    return(-y*np.log(h)-(1-y)*np.log(1-h)).mean

epoch=1000

for i in range(epoch):
    output=np.dot(x_train,weights)
    h=sigmoid(output)
    error= np.mean((h-y_train)**2)
    weights=weights-alpha*(1/m)*np.dot(np.transpose(x_train),(h-y_train))

op= np.dot(x_test, weights)
op= sigmoid(op)

c=0;
for i in range(op.shape[0]):
    if op[i][0]>0.91:
       op[i][0]=1;
    else:
        op[i][0]=0;
    if op[i][0]==y_test[i][0]:
        c=c+1;
print(c/op.shape[0])
#print(weights)

"""weights=
 
[[-1.62412570e+01]
 [-4.02971204e+00]
 [ 2.38714018e-01]
 [ 2.13321575e-01]
 [-7.75735250e+01]
 [ 3.32802032e+01]
 [ 2.61830740e+01]
 [ 9.54214733e-01]
 [ 6.61481527e+01]
 [-4.94906468e+00]
 [ 2.05468807e+01]
 [ 1.94636234e+01]
 [ 6.12277069e+00]
 [ 1.39562293e+01]
 [ 2.23237864e+00]
 [ 7.86560361e+00]
 [-2.51369989e+00]
 [ 3.79242002e+00]
 [ 7.36955936e-01]
 [ 1.49916989e+00]
 [-2.39615058e+00]
 [ 2.85633257e-01]
 [-3.73039866e+00]
 [-2.59096583e+00]
 [-5.67010236e+00]
 [-2.48870857e+00]
 [-3.09063780e-02]
 [-2.01693842e+00]]
>>> 




 these weights gave 95% effeciency"""


