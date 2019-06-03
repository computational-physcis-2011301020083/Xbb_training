import argparse,math,os,glob,h5py
import numpy as np
import keras
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
parser.add_argument("--model", dest='model',  default="", help="model")
args = parser.parse_args()

model_file=args.model
test_data=args.path
model = keras.models.load_model(model_file)
load_file=h5py.File(test_data)
data=load_file.get("test/data")
y=load_file.get("test/y")
w=load_file.get("test/w")
predictions = model.predict(data)
save_f = h5py.File("predict.h5", 'w')
save_f.create_dataset("y",data=y[:,1])
save_f.create_dataset("score",data=predictions[:,1])
save_f.create_dataset("w",data=w)



#print len(predictions[:,1][predictions[:,1]>=0.5])
#print y[:,1]>0
#print y[:,1][y[:,1]>0].sum()



