from __future__ import print_function
import numpy as np
import time
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.applications import *


x_test = np.loadtxt('encoded_seq')
x_test = x_test.reshape(-1,400, 4)
y_test = np.loadtxt('y_label')
y_true = y_test
y_test = np_utils.to_categorical(y_test, num_classes=3)


model = load_model('CNN.h5')
    
acceptor_model = load_model('acceptor_dis.h5')
donor_model = load_model('donor_dis.h5')
print(model.summary())
loss,accuracy = model.evaluate(x_test,y_test)
print('testing accuracy: {}'.format(accuracy))

predict = model.predict_classes(x_test).astype('int')

positive=0
positive_predict=0
true_positive=0

acc_true=0
acc_predict=0
acc_predict_true=0

don_true=0
don_predict=0
don_predict_true=0

for i in range(len(y_true)):
    if predict[i]!=2:
        test = x_test[i].reshape(-1,400,4)
        if predict[i]==0:
            check = acceptor_model.predict_classes(test).astype('int')
            print(check)
            if (check == 1):
                predict[i]=2
        else:
            check = donor_model.predict_classes(test).astype('int')
            if (check == 1):
                predict[i]=2

for i in range(len(y_true)):
    if y_true[i]!=2:
        positive = positive+1
        print(i,end=' ')
        print(y_true[i],end='')
        print(':',end='')
        print(predict[i])

        if y_true[i]==0:
            acc_true = acc_true+1
        else:
            don_true = don_true+1

    if predict[i]!=2:
        positive_predict = positive_predict+1
        if predict[i]==0:
            acc_predict = acc_predict+1
        else:
            don_predict = don_predict+1

    if y_true[i]!=2 and predict[i]!=2:
        true_positive = true_positive+1
        if y_true[i]==0 and predict[i]==0:
            acc_predict_true = acc_predict_true+1
        else:
            don_predict_true = don_predict_true+1


print('# of real positive is %f'%positive)
print('# of predicted positive is %f'%positive_predict)
print('# of true positive is %f'%true_positive)


print('# of real positive acc is %f'%acc_true)
print('# of predicted positive acc is %f'%acc_predict)
print('# of true positive acc is %f'%acc_predict_true)

print('# of real positive don is %f'%don_true)
print('# of predicted positive don is %f'%don_predict)
print('# of true positive don is %f'%don_predict_true)
