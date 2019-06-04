import argparse,math,os,glob,h5py
import numpy as np
import keras
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
parser.add_argument("--model", dest='model',  default="", help="model")
args = parser.parse_args()


#load mass and pt
def load_mass(data,style):
  load_f=h5py.File(data)
  load_fat=load_f.get("fat_jet/"+style)
  print load_fat[:,2].shape
  return load_fat[:,2]
 
#dijet: (865594,)
#signal: (212821,)
#top: (529776,)

files=sorted(glob.glob("SplitData/*.h5"))
bg_m=load_mass(files[0],"test")
signal_m=load_mass(files[1],"test")
top_m=load_mass(files[2],"test")
mass=np.hstack((bg_m,signal_m,top_m))


model_file=args.model
test_data=args.path
model = keras.models.load_model(model_file)
load_file=h5py.File(test_data)
data=load_file.get("test/data")
y=load_file.get("test/y")
w=load_file.get("test/w")
predictions = model.predict(data)
y=y[:,1]
score=predictions[:,1]

save_f = h5py.File("predict1.h5", 'w')
y=np.reshape(y,(y.shape[0],1))
score=np.reshape(score,(y.shape[0],1))
w=np.reshape(w,(y.shape[0],1))
mass=np.reshape(mass,(y.shape[0],1))
pre=np.hstack((y,score,w,mass))
save_f.create_dataset("pre",data=pre)




