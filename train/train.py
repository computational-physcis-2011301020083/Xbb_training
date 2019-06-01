import argparse,math,os,glob,h5py
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()
import numpy as np

import tensorflow as tf
from keras import backend as K

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


#input preparing
def load_meanstd(data,feature):
  meanstd_path="MeanStd/mean_std.h5"
  meanstd_file=h5py.File(meanstd_path,"r")
  mean_vector=meanstd_file.get(feature+"/mean")
  std_vector=meanstd_file.get(feature+"/std")
  mean_vector=np.reshape(mean_vector,(1,mean_vector.shape[0]))
  std_vector=np.reshape(std_vector,(1,std_vector.shape[0]))
  #print data
  #print mean_vector
  #print std_vector
  #std_vector[std_vector == 0] = 1 
  return (data-mean_vector)/std_vector

#feature_names = ['fat_jet', 'subjet_VR_1', 'subjet_VR_2', 'subjet_VR_3']
def load_Data(data,style):
  load_f=h5py.File(data)
  #print data
  load_fat=load_f.get("fat_jet/"+style)
  #print load_fat[1:10,:]
  data_fat=load_meanstd(load_fat,"fat_jet")
  load_VR1=load_f.get("subjet_VR_1/"+style)
  data_VR1=load_meanstd(load_VR1,"subjet_VR_1")
  load_VR2=load_f.get("subjet_VR_2/"+style)
  data_VR2=load_meanstd(load_VR2,"subjet_VR_2")
  load_VR3=load_f.get("subjet_VR_3/"+style)
  data_VR3=load_meanstd(load_VR3,"subjet_VR_3")
  load_data=np.hstack((data_fat,data_VR1,data_VR2,data_VR3))
  if "Hbb" in data:
    y=np.full((load_data.shape[0],),1,dtype=int)
    load_y=keras.utils.to_categorical(y, num_classes=3)
  if "Top" in data:
    y=np.full((load_data.shape[0],),2,dtype=int)
    load_y=keras.utils.to_categorical(y, num_classes=3)
  if "Dijets" in data:
    y=np.full((load_data.shape[0],),0,dtype=int)
    load_y=keras.utils.to_categorical(y, num_classes=3)
  weight=load_f.get("weight/"+style)
  #print weight
  load_w=weight/np.sum(weight)
  return [load_data,load_y,load_w]


files=sorted(glob.glob(args.path+"/*.h5"))
train_bg_data,train_bg_y,train_bg_w=load_Data(files[0],"train")
train_signal_data,train_signal_y,train_signal_w=load_Data(files[1],"train")
train_top_data,train_top_y,train_top_w=load_Data(files[2],"train")
train_data=np.vstack((train_bg_data,train_signal_data,train_top_data))
train_y=np.vstack((train_bg_y,train_signal_y,train_top_y))
#print train_bg_w,train_signal_w,train_top_w
train_w=np.hstack((train_bg_w,train_signal_w,train_top_w))

print train_data.shape
print train_y.shape
print train_w.shape

valid_bg_data,valid_bg_y,valid_bg_w=load_Data(files[0],"valid")
valid_signal_data,valid_signal_y,valid_signal_w=load_Data(files[1],"valid")
valid_top_data,valid_top_y,valid_top_w=load_Data(files[2],"valid")
valid_data=np.vstack((valid_bg_data,valid_signal_data,valid_top_data))
valid_y=np.vstack((valid_bg_y,valid_signal_y,valid_top_y))
#print valid_bg_w,valid_signal_w,valid_top_w
valid_w=np.hstack((valid_bg_w,valid_signal_w,valid_top_w))


#training
params={'num_layers': 6,'num_units': 250,'activation_type': 'relu','dropout_strength': 0.2,'learning_rate': 0.01,'momentum': 0.2,'lr_decay': 0.00001,'epochs': 100,'batch_norm': True,'output_size': 3,}

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


num_train_samples =4824574  # this is required since we are using generators
num_valid_samples =2573621 
train_block_size = 100000
valid_block_size = 100000
batch_size = 100

# Create new model.
model = define_model(params)
model_name = "XbbScore"


# Train model. 
initial_epoch = 0

# Callbacks
save_path = "./"
save_best = keras.callbacks.ModelCheckpoint(filepath=save_path+"models/" + model_name + "_best", monitor='val_loss', verbose=0, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
csv_logger = keras.callbacks.CSVLogger(save_path+'logs/' + model_name + '.log')
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [save_best, early_stopping, csv_logger, reduce_lr_on_plateau]  # This may need to be updated

history = model.fit(x=train_data,y=train_y,sample_weight=train_w,
		steps_per_epoch=(num_train_samples/batch_size),
		validation_data=(valid_data,valid_y,valid_w),
		validation_steps = (num_valid_samples / batch_size),
		initial_epoch = initial_epoch,
		callbacks=callbacks,
		epochs = params['epochs'] + initial_epoch,
		#verbose=2
		)









