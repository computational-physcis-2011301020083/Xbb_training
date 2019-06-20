#import talos as ta
from talos import Scan
import h5py
import numpy as np
import array
import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adam
from talos.model.normalizers import lr_normalizer
from talos.model.layers import hidden_layers


train_file=h5py.File("train.h5","r")
train=train_file.get("train")
valid=train_file.get("valid")
train_data=train[:,6:56]
train_y=train[:,0:3]
train_w=train[:,3]
valid_data=valid[:,6:56]
valid_y=valid[:,0:3]
valid_w=valid[:,3]

haper={'first_neuron':[32,64,96,128],'lr':[0.0001,0.001,0.01,0.1],'hidden_layers':[3,6,9],'batch_size':[1,100,1000,10000],'epochs':[40],'dropout':[0,0.25],'optimizer':[Adam],'losses':['categorical_crossentropy'],'activation':['relu'],'last_activation':['sigmoid'],'batchnorm':[True],'shuffle':[True],'decay':[0.00001,0.0001,0.001,0.01],'shapes':['brick']}

def hbb_model(x_train,y_train,x_val,y_val,param):
    model=Sequential()
    model.add(Dense(param['first_neuron'],input_dim=x_train.shape[1],activation=param['activation'],kernel_initializer='orthogonal'))
    model.add(Dropout(param['dropout']))
    hidden_layers(model,param,1)
    model.add(Dense(y_train.shape[1],activation='softmax',kernel_initializer='orthogonal'))
    opt=keras.optimizers.Adam(lr=param['lr'], decay=param['decay'])
    model.compile(optimizer=opt,loss=param['losses'],metrics=['acc'])
    out=model.fit(x_train,y_train,batch_size=param['batch_size'],epochs=param['epochs'],verbose=0,validation_data=[x_val,y_val])
    return out,model

h=Scan(x=train_data,y=train_y,x_val=valid_data,y_val=valid_y,params=haper,dataset_name='Hbb_optimization',experiment_no='1',model=hbb_model,grid_downsample=0.5)





