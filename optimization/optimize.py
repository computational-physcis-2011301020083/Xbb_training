import argparse
import os,glob,h5py,ROOT,shutil
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--layers", dest='layers',  default="", help="layers")
parser.add_argument("--nodes", dest='nodes',  default="", help="nodes")
parser.add_argument("--dropout", dest='dropout',  default="", help="dropout")
parser.add_argument("--rate", dest='rate',  default="", help="rate")
parser.add_argument("--decay", dest='decay',  default="", help="decay")
args = parser.parse_args()
import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CONFIG = tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=False, allow_soft_placement=False) 
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
K.set_session(sess)

print "************ PHASE LOAD FILE *****************"
train_file=h5py.File("../DataVRGhost/PrepareDataStand/train.h5","r")
train=train_file.get("train")
valid=train_file.get("valid")
train_data=train[:,6:56]
train_y=train[:,0:3]
train_w=train[:,3]
valid=train_file.get("valid")
valid_data=valid[:,6:56]
valid_y=valid[:,0:3]
valid_w=valid[:,3]

def define_model(params):
    inputs=Input(shape=(50, ), name='Julian')
    concatenated_inputs =inputs 
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
    adm = Adam(lr=params['learning_rate'], decay=params['lr_decay'])
    model.compile(loss='categorical_crossentropy', optimizer=adm)
    return model

print "************ PHASE TRAINING *****************"
params={'num_layers': int(args.layers),'num_units': int(args.nodes),'activation_type': 'relu','dropout_strength': float(args.dropout),'learning_rate': float(args.rate),'lr_decay': float(args.decay),'epochs': 1,'batch_norm': True,'output_size': 3}
model = define_model(params)
model_name = "Arc"+"_HiddenLayer"+args.layers+"_Nodes"+args.nodes+"_DropOut"+args.dropout.replace('.','p')+"_Rate"+args.rate.replace('.','p')+"_Decay"+args.decay.replace('.','p')
num_train_samples =3000000  
num_valid_samples =900000
save_path = "./"
save_best = keras.callbacks.ModelCheckpoint(filepath=save_path + model_name + "_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
csv_logger = keras.callbacks.CSVLogger(save_path + model_name + '.log')
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [save_best, early_stopping, csv_logger, reduce_lr_on_plateau] 
history = model.fit(x=train_data,y=train_y,sample_weight=train_w,validation_data=(valid_data,valid_y,valid_w),callbacks=callbacks,epochs = params['epochs'])

print "************ PHASE MAKE PREDICTIONS *****************"
log_file=model_name + '.log'
model_file=model_name + "_best.h5"
model_pre = keras.models.load_model(model_file)
load_file=h5py.File("../DataVRGhost/PrepareDataStand/train.h5")
test=load_file.get("test")
data=test[:,6:56]
y=test[:,0:3]
w=test[:,3:7]
predictions = model_pre.predict(data)
score=predictions[:,1]
prediction_file="prediction_"+model_file
save_f = h5py.File(prediction_file, 'w')
score=np.reshape(score,(y.shape[0],1))
predict=np.hstack((y,score,w))
save_f.create_dataset("predict",data=predict)

save_f.close()
import tables
tables.file._open_files.close_all()

print "************ PHASE SAVE FIGURES *****************"
os.system("python roc.py --path "+prediction_file+" --name "+model_name+" --type Dijets")
os.system("python roc.py --path "+prediction_file+" --name "+model_name+" --type Top")
os.system("python jetmass.py --path "+prediction_file+" --name "+model_name+" --type Dijets")
os.system("python jetmass.py --path "+prediction_file+" --name "+model_name+" --type Top")

shutil.move(model_file,"model/")
shutil.move(prediction_file,"model/")
shutil.move(log_file,"model/")
shutil.move("Roc_Dijets_"+model_name+".pdf","figures/")
shutil.move("Roc_Top_"+model_name+".pdf","figures/")
shutil.move("Jetmass_Dijets_"+model_name+".pdf","figures/")
shutil.move("Jetmass_Top_"+model_name+".pdf","figures/")



