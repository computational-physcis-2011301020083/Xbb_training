import argparse,math,os,glob,h5py
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()
import numpy as np

import tensorflow as tf
from keras import backend as K
'''
# Creates a graph.
c = []
for d in ['/device:GPU:0', '/device:GPU:1']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))

with tf.device('/cpu:0'):
  sum = tf.add_n(c)

# Creates a session with log_device_placement set to True.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
'''

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

train_file=h5py.File("trainfloat16.h5","r")
train_data=train_file.get("train/data")
train_y=train_file.get("train/y")
train_w=train_file.get("train/w") #[:,0]
#train_y=np.reshape(train_y,(train_y.shape[0],1))
train_w=np.reshape(train_w,(train_w.shape[0],1))
valid_data=train_file.get("valid/data")
valid_y=train_file.get("valid/y")
valid_w=train_file.get("valid/w") #[:,0]
#valid_y=np.reshape(valid_y,(valid_y.shape[0],1))
valid_w=np.reshape(valid_w,(valid_w.shape[0],1))
Train_data=np.hstack((train_y,train_data,train_w))
Valid_data=np.hstack((valid_y,valid_data,valid_w))
Train_data=np.reshape(Train_data,(Train_data.shape[0],Train_data.shape[1]))
Valid_data=np.reshape(Valid_data,(Valid_data.shape[0],Valid_data.shape[1]))

def sample_generoator(data,style,n):
    if style=="train":
        dijet_samples=int(2985000/n)
        top_samples=int(1589329/n)
        signal_samples=int(638463/n)
    if style=="valid":
        dijet_samples=int(994997/n)
        top_samples=int(847761/n)
        signal_samples=int(340288/n)
    while True:      
        for i in range(n):       
            signal_block_index=range(i*signal_samples,(i+1)*signal_samples)
            dijet_block_index=range(i*dijet_samples,(i+1)*dijet_samples)
            top_block_index=range(i*top_samples,(i+1)*top_samples)
            signal_block=np.take(data[data[:,1]==1],signal_block_index,axis=0)
            dijet_block=np.take(data[data[:,0]==1],dijet_block_index,axis=0)
            top_block=np.take(data[data[:,2]==1],top_block_index,axis=0)
            data_block=np.vstack((signal_block,dijet_block,top_block))
            yield [data_block[:,3:134],data_block[:,0:3],np.reshape(data_block[:,134:135],(data_block[:,134:135].shape[0],))]
            #print data_block.shape
            
def gen(data,style,n):
    return sample_generoator(data,style,n)
    
#gtrain=sample_generoator(Train_data,"train")
#gvalid=sample_generoator(Valid_data,"valid"

#training
params={'num_layers': 6,'num_units': 250,'activation_type': 'relu','dropout_strength': 0.2,'learning_rate': 0.01,'momentum': 0.2,'lr_decay': 0.00001,'epochs': 100,'batch_norm': True,'output_size': 3}
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


num_train_samples =5212792  # this is required since we are using generators
num_valid_samples =2183046 
train_block_size = 100000
valid_block_size = 100000
#batch_size = 1000000
batch_size = 100

gtrain=gen(Train_data,"train",batch_size)
gvalid=gen(Valid_data,"valid",batch_size)

# Create new model.
model = define_model(params)
model_name = "XbbScore_Julian"


# Train model. 
initial_epoch = 0

# Callbacks
save_path = "./"
save_best = keras.callbacks.ModelCheckpoint(filepath=save_path+ model_name + "_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
csv_logger = keras.callbacks.CSVLogger(save_path + model_name + '.log')
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [save_best, early_stopping, csv_logger, reduce_lr_on_plateau]  # This may need to be updated

#history = model.fit(x=train_data,y=train_y,sample_weight=train_w,steps_per_epoch=(num_train_samples/batch_size),validation_data=(valid_data,valid_y,valid_w),validation_steps = (num_valid_samples / batch_size),initial_epoch = initial_epoch,callbacks=callbacks,epochs = params['epochs'] + initial_epoch)
history = model.fit_generator(gtrain, 
                        steps_per_epoch=(num_train_samples/batch_size),
                        validation_data = gvalid, 
                        validation_steps = (num_valid_samples / batch_size),
                        initial_epoch = initial_epoch,
                        callbacks=callbacks,
                        epochs = params['epochs'] + initial_epoch,
                        #class_weight=None
                        #verbose=2
                        )
