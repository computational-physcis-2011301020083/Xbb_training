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
'''
test_index=np.random.choice(w.shape[0],200000,replace=False)
data=np.take(data,test_index,axis=0)
y=np.take(y,test_index,axis=0)
w=np.take(w,test_index,axis=0)
'''

predictions = model.predict(data)
#y=y[:,1]
score=predictions[:,1]
save_f = h5py.File("PrepareData/prediction.h5", 'w')

#y=np.reshape(y,(y.shape[0],1))
score=np.reshape(score,(y.shape[0],1))
predict=np.hstack((y,score,w))

save_f.create_dataset("predict",data=predict)
print predict.shape



