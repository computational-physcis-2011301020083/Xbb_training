import argparse,math,os,glob,h5py
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()
import numpy as np
import tensorflow as tf
from keras import backend as K

'''
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False) 
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
K.set_session(sess)
'''

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

#train_file=h5py.File("train.h5","r")
train_file=h5py.File("trainfloat16.h5","r")
train_data=train_file.get("train/data")
train_y=train_file.get("train/y")
train_w=train_file.get("train/w") #[:,0]
valid_data=train_file.get("valid/data")
valid_y=train_file.get("valid/y")
valid_w=train_file.get("valid/w") #[:,0]


'''
train_index=np.random.choice(5212792,200000,replace=False)
valid_index=np.random.choice(2183046,100000,replace=False)
train_data=np.take(train_data,train_index,axis=0)
valid_data=np.take(valid_data,valid_index,axis=0)
train_y=np.take(train_y,train_index,axis=0)
valid_y=np.take(valid_y,valid_index,axis=0)
train_w=np.take(train_w,train_index,axis=0)
valid_w=np.take(valid_w,valid_index,axis=0)

'''

#training
params={'num_layers': 6,'num_units': 250,'activation_type': 'relu','dropout_strength': 0.2,'learning_rate': 0.01,'momentum': 0.2,'lr_decay': 0.00001,'epochs': 1,'batch_norm': True,'output_size': 3}
#params={'num_layers': 3,'num_units': 32,'activation_type': 'relu','dropout_strength': 0.2,'learning_rate': 0.01,'momentum': 0.2,'lr_decay': 0.00001,'epochs': 1,'batch_norm': True,'output_size': 3}

def define_model(params):
    inputs=Input(shape=(131, ), name='Julian')
    concatenated_inputs =inputs  #keras.layers.concatenate(inputs)  #Input(shape=(131, ), name='Julian')
    for i in range(params['num_layers']):
        if i==0:
            x = Dense(params['num_units'], kernel_initializer='orthogonal')(concatenated_inputs)
            if params['batch_norm']:
               x = BatchNormalization()(x)
            x = Activation(params['activation_type'])(x)
            if params['dropout_strength'] > 0:
               x = Dropout(params['dropout_strength'])(x)
        else:
            x = Dense(params['num_units'], kernel_initializer='orthogonal')(x)
            if params['batch_norm']:
               x = BatchNormalization()(x)
            x = Activation(params['activation_type'])(x)
            if params['dropout_strength'] > 0:
               x = Dropout(params['dropout_strength'])(x)

    predictions = Dense(params['output_size'], activation='softmax', kernel_initializer='orthogonal')(x)
    model = Model(inputs=inputs, outputs=predictions)
    sgd = SGD(lr=params['learning_rate'], decay=params['lr_decay'], momentum=params['momentum'], nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


#num_train_samples =200000  # this is required since we are using generators
#num_valid_samples =100000

num_train_samples =5212792  # this is required since we are using generators
num_valid_samples =2183046
train_block_size = 100000
valid_block_size = 100000
#batch_size = 1000000
batch_size = 100

# Create new model.
model = define_model(params)
model_name = "XbbScore_Julian"


# Train model. 
initial_epoch = 0

# Callbacks
save_path = "./"
save_best = keras.callbacks.ModelCheckpoint(filepath=save_path + model_name + "_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
csv_logger = keras.callbacks.CSVLogger(save_path + model_name + '.log')
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [save_best, early_stopping, csv_logger, reduce_lr_on_plateau]  # This may need to be updated

history = model.fit(x=train_data,y=train_y,sample_weight=train_w,steps_per_epoch=(num_train_samples/batch_size),validation_data=(valid_data,valid_y,valid_w),validation_steps = (num_valid_samples / batch_size),initial_epoch = initial_epoch,callbacks=callbacks,epochs = params['epochs'] + initial_epoch)

