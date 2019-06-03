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

outname="train.h5"
save_f = h5py.File("PrepareData/"+outname, 'w')

files=sorted(glob.glob(args.path+"/*.h5"))
train_bg_data,train_bg_y,train_bg_w=load_Data(files[0],"train")
train_signal_data,train_signal_y,train_signal_w=load_Data(files[1],"train")
train_top_data,train_top_y,train_top_w=load_Data(files[2],"train")
train_data=np.vstack((train_bg_data,train_signal_data,train_top_data))
train_y=np.vstack((train_bg_y,train_signal_y,train_top_y))
#print train_bg_w,train_signal_w,train_top_w
train_w=np.hstack((train_bg_w,train_signal_w,train_top_w))

save_f.create_dataset("train/data",data=train_data)
save_f.create_dataset("train/y",data=train_y)
save_f.create_dataset("train/w",data=train_w)
#print train_data.shape
#print train_y.shape
#print train_w.shape

valid_bg_data,valid_bg_y,valid_bg_w=load_Data(files[0],"valid")
valid_signal_data,valid_signal_y,valid_signal_w=load_Data(files[1],"valid")
valid_top_data,valid_top_y,valid_top_w=load_Data(files[2],"valid")
valid_data=np.vstack((valid_bg_data,valid_signal_data,valid_top_data))
valid_y=np.vstack((valid_bg_y,valid_signal_y,valid_top_y))
#print valid_bg_w,valid_signal_w,valid_top_w
valid_w=np.hstack((valid_bg_w,valid_signal_w,valid_top_w))

save_f.create_dataset("valid/data",data=valid_data)
save_f.create_dataset("valid/y",data=valid_y)
save_f.create_dataset("valid/w",data=valid_w)

#print valid_data.shape
#print valid_y.shape
#print valid_w.shape

test_bg_data,test_bg_y,test_bg_w=load_Data(files[0],"test")
test_signal_data,test_signal_y,test_signal_w=load_Data(files[1],"test")
test_top_data,test_top_y,test_top_w=load_Data(files[2],"test")
test_data=np.vstack((test_bg_data,test_signal_data,test_top_data))
test_y=np.vstack((test_bg_y,test_signal_y,test_top_y))
#print test_bg_w,test_signal_w,test_top_w
test_w=np.hstack((test_bg_w,test_signal_w,test_top_w))

save_f.create_dataset("test/data",data=test_data)
save_f.create_dataset("test/y",data=test_y)
save_f.create_dataset("test/w",data=test_w)







 












