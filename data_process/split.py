import numpy as np
import pandas as pd
import glob,h5py
import argparse,math,os
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

load_f = h5py.File(args.path, 'r')
outname=args.path.split("/")[-1]
save_f = h5py.File("SplitData/"+outname, 'w')

weight=load_f.get("weight")
N=weight.shape[0]
train_index=np.random.choice(N,int(N*0.6),replace=False)
remain_index=np.delete(np.arange(0,N),train_index) 
test_index=np.random.choice(remain_index,int(N*0.2),replace=False)
valid_index=np.delete(remain_index,test_index) 

case=["train","test","valid"]
index={"train":train_index,"test":test_index,"valid":valid_index}
feature_names = ['fat_jet', 'subjet_VR_1', 'subjet_VR_2', 'subjet_VR_3', 'weight']
for i in feature_names:
    df1=load_f.get(i)
    for j in  case:
      save_f.create_dataset(i+"/"+j,data=np.take(df1,index[j],axis=0))




